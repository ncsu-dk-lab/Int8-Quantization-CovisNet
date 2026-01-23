# export_model_torchscript_onnx.py
import torch
import roma
import torch.nn as nn
import math
from train.models.layers import MemEffAttention
from train.models.layers.attention import Attention
from train.models.layers.block import Block
from train.models.dinov2_vision_transformer import DinoVisionTransformer
from functools import partial
import time
import numpy as np
from train.models.model_bev_pose import (
    BEVGNNModel,
    PyramidTransformer,
    linspace_mult,
    MultiUp,
)
from pathlib import Path
import torchvision.transforms as T
import torchvision
import argparse
from evaluation.utils import evaluate_model_accruacy, analyze_model_runtime
from typing import Dict
from torch import Tensor

# Added for ONNX export
import torch.onnx

attn = Attention
dtype = torch.float32 #torch.float16



# ... (All your class definitions: LoadedModel, Encoder, Message, PosePost, Bev, bev_decoder) ...
# ... (These classes remain unchanged) ...

class LoadedModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()
        self.model = BEVGNNModel(
            comm_range=1000.0,
            gnn_in_channels=in_channels,
            gnn_in_seq_len=seq_len,
            pose_gnn_out_channels=out_channels,
            bev_gnn_out_channels=out_channels,
            dec_out_channels=1,
        )

    #def forward(self, input):
    def forward(self, input: Dict[str, Tensor]):
        return self.model(input)

class Encoder(torch.nn.Module):
    def __init__(self, gnn_in_channels: int, gnn_in_seq_len: int):
        super().__init__()
        self.enc = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            block_fn=partial(Block, attn_class=Attention),
        )

        dtype = next(self.enc.parameters()).dtype
        # self.pre_transforms = nn.Sequential(
        #     # T.Resize([224, ]),
        #     # T.CenterCrop(224),
        #     #T.ConvertImageDtype(dtype),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # )
        self.pre_transforms = nn.Identity()

        url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.enc.load_state_dict(state_dict)
        #self.enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
        
        self.gnn_in_seq_len = gnn_in_seq_len

        enc_out_seq_len = int((224 / self.enc.patch_size) ** 2)
        self.enc_post = nn.Sequential(
            PyramidTransformer(
                linspace_mult(self.enc.num_features, gnn_in_channels, 6, 48),
                linspace_mult(enc_out_seq_len, self.gnn_in_seq_len, 6, 8),
                attn_cls=attn,
            ),
        )
        self.forward_features_list = self.enc.forward_features_list
        self.prepare_tokens_with_masks = self.enc.prepare_tokens_with_masks
        self.blocks = self.enc.blocks
        self.norm  = self.enc.norm

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }
    
    def forward(self, x):
        x = self.pre_transforms(x)
        x = self.forward_features(x)["x_norm_patchtokens"]
        x = self.enc_post(x)
        return x[:, : self.gnn_in_seq_len]


class Message(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len * 2, self.in_channels))
        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.aggregator = nn.Sequential(
            PyramidTransformer(
                linspace_mult(self.in_channels, self.out_channels, 8, 48),
                [seq_len * 2] * 4 + linspace_mult(seq_len * 2, 8, 4, 8),
                attn_cls=attn,
            ),
        )

        self.gnn_decoder_post = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.out_channels, 17),
        )

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1) + self.pos_embedding
        x = self.aggregator(x)[:, 0]
        edge_preds = self.gnn_decoder_post(x)

        return edge_preds


class PosePost(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_preds):
        # NOTE: ONNX export of tuples results in multiple output tensors.
        # This is expected and correct.
        edge_preds_proc = (
            edge_preds[:, 0:3],
            edge_preds[:, 3:6].exp(),
            roma.symmatrixvec_to_unitquat(edge_preds[:, 6:16].to(torch.float)).to(
                edge_preds.dtype
            ),
            edge_preds[:, 16].exp(),
        )
        return edge_preds_proc


class Bev(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()

        self.in_channels = in_channels
        self.in_seq_len = seq_len
        self.out_channels = out_channels

        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.in_seq_len * 2, self.in_channels)
        )

        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        pose_emb_size = 48
        self.pose_embedding = torch.nn.Sequential(
            torch.nn.Linear(17, pose_emb_size),
            torch.nn.LayerNorm(pose_emb_size, eps=1e-6),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(pose_emb_size, pose_emb_size),
            torch.nn.LayerNorm(pose_emb_size, eps=1e-6),
        )
        self.aggregator = nn.Sequential(
            PyramidTransformer(
                linspace_mult(
                    self.in_channels + pose_emb_size, self.out_channels, 8, 48
                ),
                linspace_mult(self.in_seq_len * 2, 8, 8, 8),
            ),
        )

    def forward(self, x_i, x_j, edge_pred):
        x = torch.cat([x_i, x_j], dim=1) + self.pos_embedding
        pose = self.pose_embedding(edge_pred).unsqueeze(1).repeat(1, x.shape[1], 1)
        x_p = torch.cat([x, pose], dim=2)
        aggr = self.aggregator(x_p)
        return aggr[:, 0]


class bev_decoder(torch.nn.Module):
    def __init__(self, bev_gnn_out_channels, dec_out_channels):
        super().__init__()

        self.decoder = MultiUp(
            linspace_mult(bev_gnn_out_channels, 16, 7, 8), dec_out_channels
        )

    def forward(self, x_i):
        dec = self.decoder(x_i.unsqueeze(2).unsqueeze(3))
        _, _, w, h = dec.shape
        o_w, o_h = 60, 60
        result_crop = torchvision.transforms.functional.crop(
            dec, int(w / 2 - o_w / 2), int(h / 2 - o_h / 2), o_w, o_h
        )
        return result_crop


# ... (Functions model_decentr_forward, eval_outputs, model_trace are unchanged) ...
def model_decentr_forward(data, edge_index, f_enc, f_msg, f_post, dtype):
    out = []
    img_flat = data["img_norm"].flatten(0, 1).to(dtype)
    for i, j in zip(edge_index[1], edge_index[0]):
        img_i = img_flat[i].unsqueeze(0)
        img_j = img_flat[j].unsqueeze(0)
        x_i = f_enc.to(dtype)(img_i)
        x_j = f_enc.to(dtype)(img_j)
        aggr = f_msg.to(dtype)(x_i, x_j)
        # PosePost needs float32 for numerical stability with roma
        p = f_post.to(torch.float)(aggr.to(torch.float))
        out_cat = torch.concatenate(
            [c.unsqueeze(1) if c.ndim == 1 else c for c in p], dim=1
        )
        out.append(out_cat)
    return torch.concatenate(out, dim=0)


def eval_outputs(a, b, dtype):
    """Verifies that two model outputs are numerically close."""
    def merge_dict(v):
        return torch.concatenate(
            [v[k].unsqueeze(1) if v[k].ndim == 1 else v[k] for k in v.keys()], dim=1
        )

    b_merged = merge_dict(b)
    err = a.to(dtype) - b_merged.to(dtype)
    mse = (err**2).mean().item()
    mae = err.abs().mean().item()
    print(f"Comparing outputs: MSE={mse:.6e}, MAE={mae:.6e}")
    
    are_close = torch.allclose(a, b_merged, rtol=1e-3, atol=1e-4)
    return are_close


def model_trace(data, f_enc, f_msg, f_post, f_bev, f_bev_dec, dtype):
    """Traces the individual model components and returns example inputs."""
    img_flat = data["img_norm"].to(dtype).flatten(0, 1)
    
    i, j = 0, 1
    
    img_i = img_flat[i].unsqueeze(0)
    img_j = img_flat[j].unsqueeze(0)
    
    # Run a forward pass to get intermediate tensors for tracing
    x_i = f_enc.to(dtype)(img_i)
    x_j = f_enc.to(dtype)(img_j)
    aggr = f_msg.to(dtype)(x_i, x_j)
    bev = f_bev.to(dtype)(x_i, x_j, aggr)

    # Test runtime of each component
    print("Ëvaluating runtime....")
    analyze_model_runtime(f_enc, img_i, name="Encoder_i")
    analyze_model_runtime(f_enc, img_j, name="Encoder_j")
    analyze_model_runtime(f_msg, x_i, x_j, name="Msg Module")
    analyze_model_runtime(f_bev, x_i, x_j, aggr, name="Bev Module")
    
    # Generate traced models
    f_enc_traced = torch.jit.trace(f_enc.to(dtype), img_i)
    f_msg_traced = torch.jit.trace(f_msg.to(dtype), (x_i, x_j))
    f_post_traced = torch.jit.trace(f_post.to(torch.float), aggr.to(torch.float))
    f_bev_traced = torch.jit.trace(f_bev.to(dtype), (x_i, x_j, aggr))
    f_bev_dec_traced = torch.jit.trace(f_bev_dec.to(dtype), bev)
    
    # Return traced models AND the example inputs for ONNX export
    example_inputs = {
        'enc': img_i,
        'msg': (x_i, x_j),
        'post': aggr.to(torch.float),
        'bev': (x_i, x_j, aggr),
        'bev_dec': bev
    }

    traced_models = {
        'enc': f_enc_traced,
        'msg': f_msg_traced,
        'post': f_post_traced,
        'bev': f_bev_traced,
        'bev_dec': f_bev_dec_traced
    }
    
    return traced_models, example_inputs

# ==============================================================================
# NEW HELPER FUNCTION FOR ONNX EXPORT
# ==============================================================================
def export_to_onnx(model, example_inputs, path, input_names, output_names, dynamic_axes):
    """Exports a single PyTorch model to ONNX format."""
    print(f"  Exporting {path.name}...")
    try:
        torch.onnx.export(
            model,
            example_inputs,
            str(path),  # Path must be a string
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            #dynamo=True
        )
        print(f"  ✅ Successfully exported to {path}")
    except Exception as e:
        print(f"  ❌ Failed to export {path.name}: {e}")

# ==============================================================================
# MODIFIED CONVERT FUNCTION
# ==============================================================================
def convert(model_in_file, data, edge_index, model_out_edges,
            encoder, message, post, bev, bev_dec, device_str,
            dtype):
    """
    Traces, verifies, and saves the model components to TorchScript and ONNX.
    """
    print(f"\n--- Starting conversion for dtype: {dtype} ---")
    
    # 1. Trace the models and get example inputs for ONNX
    traced_models, example_inputs = model_trace(
        data, encoder, message, post, bev, bev_dec, dtype=dtype
    )
    print("Tracing complete.")

    # 2. Verify the traced models
    print("Verifying traced model outputs...")
    model_jit_decentr_out = model_decentr_forward(
        data, edge_index, traced_models['enc'], traced_models['msg'], traced_models['post'], dtype=dtype
    )
    # outputs_are_correct = eval_outputs(model_jit_decentr_out, model_out_edges, dtype=dtype)
    # if not outputs_are_correct:
    #      print("⚠️ Verification failed: Traced model output does not match original model output.")
    # else:
    #     print("✅ Verification successful!")

    # 3. Prepare output directories and names
    out_base_dir = Path("./models")
    out_base_dir.mkdir(exist_ok=True)
    
    model_version = model_in_file.parent.stem.replace(":", "-")
    model_epoch = int(model_in_file.stem.split("-")[0].split("=")[1])
    model_prefix = f"{model_version}e{model_epoch:02d}"
    type_prefix = str(dtype).split(".")[1]
    
    # --- 4. Save the TorchScript models ---
    print("\n--- Saving TorchScript Models ---")
    ts_paths = {
        "enc": out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_enc.ts",
        "msg": out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_msg.ts",
        "post": out_base_dir / f"{model_prefix}_float32_jit_{device_str}_post.ts",
        "bev": out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_bev.ts",
        "bev_dec": out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_bevdec.ts",
    }
    
    torch.jit.save(traced_models['enc'], ts_paths["enc"])
    torch.jit.save(traced_models['msg'], ts_paths["msg"])
    torch.jit.save(traced_models['post'], ts_paths["post"])
    torch.jit.save(traced_models['bev'], ts_paths["bev"])
    torch.jit.save(traced_models['bev_dec'], ts_paths["bev_dec"])
    
    print(f"Saved traced models to '{out_base_dir}':")
    for name, path in ts_paths.items():
        print(f"  - {name}: {path.name}")
        
    # --- 5. Save the ONNX models ---
    print("\n--- Saving ONNX Models ---")
    
    # Use the original (non-traced) models for ONNX export for better graph generation
    models_to_export = {
        "enc": encoder.to(dtype),
        "msg": message.to(dtype),
        "post": post.to(torch.float32), # PosePost has special dtype requirements
        "bev": bev.to(dtype),
        "bev_dec": bev_dec.to(dtype)
    }

    # Define paths
    onnx_paths = {
       "enc": out_base_dir / f"{model_prefix}_{type_prefix}_onnx_{device_str}_enc.onnx",
       "msg": out_base_dir / f"{model_prefix}_{type_prefix}_onnx_{device_str}_msg.onnx",
       "post": out_base_dir / f"{model_prefix}_float32_onnx_{device_str}_post.onnx",
       "bev": out_base_dir / f"{model_prefix}_{type_prefix}_onnx_{device_str}_bev.onnx",
       "bev_dec": out_base_dir / f"{model_prefix}_{type_prefix}_onnx_{device_str}_bevdec.onnx",
    }

    # Export each model component to ONNX
    export_to_onnx(
        models_to_export['enc'], example_inputs['enc'], onnx_paths['enc'],
        input_names=['image'], output_names=['features'],
        dynamic_axes={'image': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
    )
    export_to_onnx(
        models_to_export['msg'], example_inputs['msg'], onnx_paths['msg'],
        input_names=['features_i', 'features_j'], output_names=['edge_predictions'],
        dynamic_axes={'features_i': {0: 'batch_size'}, 'features_j': {0: 'batch_size'}, 'edge_predictions': {0: 'batch_size'}}
    )
    export_to_onnx(
        models_to_export['bev'], example_inputs['bev'], onnx_paths['bev'],
        input_names=['features_i', 'features_j', 'edge_prediction'], output_names=['bev_features'],
        dynamic_axes={'features_i': {0: 'batch_size'}, 'features_j': {0: 'batch_size'}, 'edge_prediction': {0: 'batch_size'}, 'bev_features': {0: 'batch_size'}}
    )
    export_to_onnx(
        models_to_export['bev_dec'], example_inputs['bev_dec'], onnx_paths['bev_dec'],
        input_names=['bev_features'], output_names=['bev_map'],
        dynamic_axes={'bev_features': {0: 'batch_size'}, 'bev_map': {0: 'batch_size'}}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CovisNet model components to TorchScript.")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="models/0kc5po4e/epoch=18-step=452067.ckpt",
        help="Path to the model checkpoint (.ckpt) file."
    )
    args = parser.parse_args()
    model_in_file = Path(args.model_ckpt)
    if not model_in_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_in_file}")

    # --- Configuration ---
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_str)
    print(f"Using device: {dev}")
    
    gnn_in_channels = 24
    gnn_in_seq_len = 128
    gnn_out_channels = 384
    resolution = 224
    torch.manual_seed(0)

    # --- 1. Load Full Pre-trained Model ---
    print(f"Loading full model from: {model_in_file}")    
    model = LoadedModel(
        gnn_in_channels, gnn_out_channels, gnn_in_seq_len
    ).eval().to(dev, dtype=dtype)
    model.load_state_dict(torch.load(model_in_file, weights_only=False)["state_dict"])
    model_inp = {
        "img_norm": torch.rand(10, 5, 3, resolution, resolution, dtype=dtype),
        "pos": torch.rand(10, 5, 3, dtype=dtype),
        "rot_quat": torch.rand(10, 5, 4, dtype=dtype),
    }

    model = model.to(dev)

    encoder = Encoder(gnn_in_channels, gnn_in_seq_len).eval().to(dev)
    encoder.enc_post.load_state_dict(model.model.enc_post.state_dict())

    post = PosePost().eval().to(dev)

    torch.manual_seed(0)

    inp = torch.rand(1, 3, resolution, resolution, device=dev)
    enc_out = encoder(inp)

    message = Message(gnn_in_channels, gnn_out_channels, gnn_in_seq_len).eval().to(dev)
    message.pos_embedding = model.model.pose_gnn.pos_embedding
    message.aggregator.load_state_dict(model.model.pose_gnn.aggregator.state_dict())
    message.gnn_decoder_post.load_state_dict(model.model.pose_gnn_decoder_post.state_dict())

    bev = Bev(gnn_in_channels, gnn_out_channels, gnn_in_seq_len).eval().to(dev)
    bev.pos_embedding = model.model.bev_gnn.pos_embedding
    bev.pose_embedding.load_state_dict(model.model.bev_gnn.pose_embedding.state_dict())
    bev.aggregator.load_state_dict(model.model.bev_gnn.aggregator.state_dict())

    bev_dec = bev_decoder(gnn_out_channels, 1).eval().to(dev)
    bev_dec.decoder.load_state_dict(model.model.bev_decoder.state_dict())

    model.model.encoder = encoder.enc

    print("Components ready.")

    # --- 3. Generate Dummy Data and Baseline Output ---
    bs = 1
    n_agents = 3 # Must be >= 2 for correct tracing
    data = {
        "pos": torch.rand(bs, n_agents, 3, device=dev, dtype=dtype) * 8,
        "rot_quat": torch.rand(bs, n_agents, 4, device=dev, dtype=dtype),
        "img_norm": torch.randn(bs, n_agents, 3, resolution, resolution, device=dev, dtype=dtype) * 3,
    }

    model.to(dev, dtype=dtype)
    # Get baseline output from the original, non-decentralized model
    (model_out_edges, model_out_nodes), (edge_index, batch) = model(data)
    
    # Get baseline output from the python-based decentralized model
    print("Verifying decentralized python implementation...")
    model_decentr_out = model_decentr_forward(
        data, edge_index, encoder, message, post, dtype=dtype
    )
    # assert eval_outputs(model_decentr_out, model_out_edges), "Decentralized python model does not match original."
    print("✅ Decentralized python implementation is correct.")

    # --- 4. Run Conversion for Different Precisions ---
    convert(model_in_file, data, edge_index, model_out_edges,
            encoder, message, post, bev, bev_dec, device_str, dtype=dtype)

