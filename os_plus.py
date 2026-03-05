# export_model_torchscript_onnx.py
import torch
import roma
import torch.nn as nn
import math
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
print(f"Loading full model from: models/0kc5po4e/epoch=18-step=452067.ckpt")    
model = LoadedModel(
    gnn_in_channels, gnn_out_channels, gnn_in_seq_len
).eval().to(dev, dtype=dtype)
model.load_state_dict(torch.load("models/0kc5po4e/epoch=18-step=452067.ckpt", weights_only=False)["state_dict"])
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


import torch
import torch.nn as nn

# ----------------------------
# 1) Collect per-channel shift z on LN outputs
# ----------------------------
@torch.no_grad()
def collect_ln_channel_shift(model: nn.Module, ln_modules: dict, calib_inputs, device):
    """
    ln_modules: {name: layernorm_module}
    calib_inputs: iterable of input tensors for encoder forward, e.g. [inp1, inp2, ...]
                 each input shape [B, 3, H, W]
    returns: {name: z tensor of shape [C]}
    """
    stats = {}
    hooks = []

    for name, ln in ln_modules.items():
        C = ln.normalized_shape[0] if isinstance(ln.normalized_shape, (tuple, list)) else ln.normalized_shape
        stats[name] = {
            "min": torch.full((C,), float("inf"), device=device),
            "max": torch.full((C,), float("-inf"), device=device),
        }

        def make_hook(key):
            def hook(module, inp, out):
                # out shape usually [B, N, C]
                x = out.detach()
                x = x.reshape(-1, x.shape[-1])   # collapse all but channel dim
                cur_min = x.amin(dim=0)
                cur_max = x.amax(dim=0)
                stats[key]["min"] = torch.minimum(stats[key]["min"], cur_min)
                stats[key]["max"] = torch.maximum(stats[key]["max"], cur_max)
            return hook

        hooks.append(ln.register_forward_hook(make_hook(name)))

    model.eval()
    for x in calib_inputs:
        x = x.to(device)
        _ = model(x)

    for h in hooks:
        h.remove()

    z_dict = {}
    for name, s in stats.items():
        z = (s["min"] + s["max"]) / 2.0
        z_dict[name] = z
    return z_dict


# ----------------------------
# 2) Fold shift into LN -> Linear
# ----------------------------
@torch.no_grad()
def fold_ln_shift_into_linear(ln: nn.LayerNorm, linear: nn.Linear, z: torch.Tensor):
    """
    Makes LN output effectively become (y - z) without adding runtime ops.

    Original:
        y = LN(x)
        out = Linear(y)

    After folding:
        y' = LN'(x) = y - z
        out' = Linear'(y') = out

    This is done by:
        LN.bias  <- LN.bias - z
        Linear.bias <- Linear.bias + W @ z
    """
    assert isinstance(ln, nn.LayerNorm)
    assert isinstance(linear, nn.Linear)
    assert ln.elementwise_affine, "LayerNorm must have affine=True to fold shift."

    z = z.to(device=linear.weight.device, dtype=linear.weight.dtype)

    # 1) shift LN output by subtracting z
    if ln.bias is None:
        raise ValueError("LayerNorm has no bias; expected affine LayerNorm with bias.")
    ln.bias.data.sub_(z.to(device=ln.bias.device, dtype=ln.bias.dtype))

    # 2) compensate next Linear bias: b <- b + W @ z
    # PyTorch Linear: out = x @ W^T + b, weight shape [out_features, in_features]
    delta_b = torch.matmul(linear.weight.data, z)  # [out_features]

    if linear.bias is None:
        linear.bias = nn.Parameter(delta_b.clone())
    else:
        linear.bias.data.add_(delta_b.to(device=linear.bias.device, dtype=linear.bias.dtype))


# ----------------------------
# 3) Build safe LN -> Linear pairs for your Encoder
# ----------------------------
def get_encoder_shift_pairs(encoder: nn.Module):
    """
    Returns:
        ln_dict:   {name: ln_module}
        pair_dict: {name: (ln_module, next_linear)}
    """
    ln_dict = {}
    pair_dict = {}

    # DINOv2 encoder blocks
    for i, blk in enumerate(encoder.blocks):
        # norm1 -> attn.qkv
        if hasattr(blk, "norm1") and hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
            name = f"blocks.{i}.norm1"
            ln_dict[name] = blk.norm1
            pair_dict[name] = (blk.norm1, blk.attn.qkv)

        # norm2 -> mlp.fc1
        if hasattr(blk, "norm2") and hasattr(blk, "mlp") and hasattr(blk.mlp, "fc1"):
            name = f"blocks.{i}.norm2"
            ln_dict[name] = blk.norm2
            pair_dict[name] = (blk.norm2, blk.mlp.fc1)

    return ln_dict, pair_dict


# ----------------------------
# 4) One-shot apply basic channel shifting to encoder
# ----------------------------
@torch.no_grad()
def apply_basic_os_channel_shift_to_encoder(encoder: nn.Module, calib_inputs, device):
    """
    encoder: your Encoder(...) object
    calib_inputs: iterable of image tensors, e.g. [inp1, inp2, ...]
    """
    ln_dict, pair_dict = get_encoder_shift_pairs(encoder)

    # collect z on LN outputs
    z_dict = collect_ln_channel_shift(encoder, ln_dict, calib_inputs, device)

    # fold shifts
    for name, z in z_dict.items():
        ln, linear = pair_dict[name]
        fold_ln_shift_into_linear(ln, linear, z)
        print(f"[shifted] {name} -> folded into {linear.__class__.__name__}, z.shape={tuple(z.shape)}")

    return z_dict



# Example calibration inputs
calib_inputs = [
    torch.rand(1, 3, resolution, resolution, device=dev, dtype=dtype)
    for _ in range(8)
]

# Apply basic OS channel shifting to DINOv2 encoder only
z_dict = apply_basic_os_channel_shift_to_encoder(
    encoder=encoder,
    calib_inputs=calib_inputs,
    device=dev,
)