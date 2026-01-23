import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import modelopt.torch.quantization as mtq

from .export_model_torchscript import LoadedModel, Encoder, Message, Bev, BevDecoder, PosePost

def load_all_module():
    model_in_file = Path("models/0kc5po4e/epoch=18-step=452067.ckpt")
    device_str = "cuda"
    dev = torch.device(device_str)
    gnn_in_channels = 24
    gnn_in_seq_len = 128
    gnn_out_channels = 384
    resolution = 224

    model = LoadedModel(
        gnn_in_channels, gnn_out_channels, gnn_in_seq_len
    ).eval()  # .to(dev)
    model.load_state_dict(torch.load(model_in_file, weights_only=False)["state_dict"])

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

    bev_dec = BevDecoder(gnn_out_channels, 1).eval().to(dev)
    bev_dec.decoder.load_state_dict(model.model.bev_decoder.state_dict())
    
    model.model.encoder = encoder.enc
    return model, encoder, message, bev, bev_dec, post

# ============================
# Calibration dataset
# ============================
class NpzCalibDataset(Dataset):
    """
    Generic NPZ-backed dataset that can handle:
      - single-input modules (one array in npz)
      - multi-input modules (x_i, x_j, edge_pred, ...)
    """
    def __init__(
        self,
        npz_path: str,
        keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data = np.load(npz_path)
        # keys to use for this module
        if keys is None:
            self.keys = list(self.data.keys())
        else:
            self.keys = keys

        if len(self.keys) == 0:
            raise ValueError(f"No arrays found in {npz_path}")

        # All arrays must have same leading dimension
        first_array = self.data[self.keys[0]]
        self._len = first_array.shape[0]
        for k in self.keys[1:]:
            if self.data[k].shape[0] != self._len:
                raise ValueError(
                    f"Array '{k}' has length {self.data[k].shape[0]} "
                    f"but '{self.keys[0]}' has length {self._len}"
                )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if len(self.keys) == 1:
            # Single input: return tensor directly
            arr = self.data[self.keys[0]][idx]
            return torch.from_numpy(arr)
        else:
            # Multiple inputs: return dict of tensors
            out = {}
            for k in self.keys:
                arr = self.data[k][idx]
                out[k] = torch.from_numpy(arr)
            return out


def make_calib_loader(npz_path, keys=None, batch_size=8):
    ds = NpzCalibDataset(npz_path, keys=keys)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ============================
# ONNX export helper
# ============================
def export_to_onnx(model, example_inputs, path, input_names, output_names, dynamic_axes):
    """
    Exports a single PyTorch model to ONNX format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Exporting {path.name}...")
    print(f"\n=== ONNX EXPORT ===")
    print(f"cwd        : {os.getcwd()}")
    print(f"target path: {path!r}")
    print(f"abs path   : {path.resolve()!r}")

    try:
        torch.onnx.export(
            model,
            example_inputs,
            str(path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"  ✅ Successfully exported to {path}")
    except Exception as e:
        print(f"  ❌ Failed to export {path.name}: {e}")
    finally:
        print(f"  exists? {path.exists()}")


# ============================
# Module-specific forward_loops
# ============================
def make_forward_loop_encoder(calib_loader, device):
    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            # batch: [B,3,H,W]
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
                _ = m(batch)
            else:
                # In case npz was saved with a dict-like structure
                # and you passed keys, fall back to one of them.
                if isinstance(batch, dict):
                    # pick the first key
                    k0 = next(iter(batch.keys()))
                    x = batch[k0].to(device, non_blocking=True)
                    _ = m(x)
                else:
                    raise TypeError(f"Unexpected batch type for encoder: {type(batch)}")
    return forward_loop


def make_forward_loop_message(calib_loader, device):
    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            # batch should be dict {"x_i": ..., "x_j": ...}
            if not isinstance(batch, dict):
                raise TypeError("Message calib npz must be loaded as dict with keys 'x_i', 'x_j'")
            x_i = batch["x_i"].to(device, non_blocking=True)
            x_j = batch["x_j"].to(device, non_blocking=True)
            _ = m(x_i, x_j)
    return forward_loop


def make_forward_loop_bev(calib_loader, device):
    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            # batch: dict {"x_i": ..., "x_j": ..., "edge_pred": ...}
            if not isinstance(batch, dict):
                raise TypeError("Bev calib npz must be dict with keys 'x_i', 'x_j', 'edge_pred'")
            x_i = batch["x_i"].to(device, non_blocking=True)
            x_j = batch["x_j"].to(device, non_blocking=True)
            edge_pred = batch["edge_preds"].to(device, non_blocking=True)
            _ = m(x_i, x_j, edge_pred)
    return forward_loop


def make_forward_loop_bev_dec(calib_loader, device):
    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            # Single input for bev_dec: bev_features
            if isinstance(batch, torch.Tensor):
                bev_feat = batch.to(device, non_blocking=True)
            elif isinstance(batch, dict):
                # in case you stored under a name like "bev_features"
                k0 = next(iter(batch.keys()))
                bev_feat = batch[k0].to(device, non_blocking=True)
            else:
                raise TypeError(f"Unexpected batch type for bev_dec: {type(batch)}")
            _ = m(bev_feat)
    return forward_loop


def make_forward_loop_post(calib_loader, device):
    """
    Optional: if you really want to SmoothQuant PosePost.
    NOTE: your earlier pipeline kept PosePost in FP32 because of roma.
    """
    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            if isinstance(batch, torch.Tensor):
                edge_pred = batch.to(device, non_blocking=True)
            elif isinstance(batch, dict):
                k0 = next(iter(batch.keys()))
                edge_pred = batch[k0].to(device, non_blocking=True)
            else:
                raise TypeError(f"Unexpected batch type for post: {type(batch)}")
            _ = m(edge_pred)
    return forward_loop


# ============================
# Main: quantize + export
# ============================
def main():
    # --------------------
    # Config / paths
    # --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

    resolution = 224
    gnn_in_channels = 24
    gnn_in_seq_len = 128
    gnn_out_channels = 384

    # Replace these with your actual calib npz paths
    CALIB_DIR = "./calib"
    calib_paths = {
        "enc": os.path.join(CALIB_DIR, "calib_encoder_inputs.npz"),
        "msg": os.path.join(CALIB_DIR, "calib_message_inputs.npz"),
        "bev": os.path.join(CALIB_DIR, "calib_bev_inputs.npz"),
        "bev_dec": os.path.join(CALIB_DIR, "calib_bevdecoder_inputs.npz"),
        # "post": os.path.join(CALIB_DIR, "calib_post_inputs.npz"),  # optional
    }

    out_dir = Path("./models")
    out_dir.mkdir(exist_ok=True)

    # Quantization config
    config = mtq.INT8_SMOOTHQUANT_CFG

    # --------------------
    # Load modules
    # --------------------
    # You already have this:
    #   _, encoder, message, bev, bev_dec, post = load_all_module()
    _, encoder, message, bev, bev_dec, post = load_all_module()

    encoder = encoder.to(dev).eval()
    message = message.to(dev).eval()
    bev = bev.to(dev).eval()
    bev_dec = bev_dec.to(dev).eval()
    # post = post.to(dev).eval()  # if you want to quantize it

    # --------------------
    # Build calib loaders
    # --------------------
    # Encoder: single input, assume npz has either one array, or a key like "img" or "img_norm".
    enc_loader = make_calib_loader(
        calib_paths["enc"],
        keys=["imgs"],         # or ["img_norm"] if you stored under specific key
        batch_size=8,
    )

    # Message: x_i, x_j
    # msg_loader = make_calib_loader(
    #     calib_paths["msg"],
    #     keys=["x_i", "x_j"],
    #     batch_size=8,
    # )

    # # Bev: x_i, x_j, edge_pred
    # bev_loader = make_calib_loader(
    #     calib_paths["bev"],
    #     keys=["x_i", "x_j", "edge_preds"],
    #     batch_size=8,
    # )

    # # Bev decoder: bev_features (or whatever key you used)
    # bev_dec_loader = make_calib_loader(
    #     calib_paths["bev_dec"],
    #     keys=["bev_feats"],  # or ["bev_features"]
    #     batch_size=8,
    # )

    # --------------------
    # SmoothQuant + INT8 per module
    # --------------------
    print("\n=== Quantizing encoder with SmoothQuant INT8 ===")
    
    enc_forward_loop = make_forward_loop_encoder(enc_loader, dev)
    encoder_int8 = mtq.quantize(encoder, config, enc_forward_loop)
    mtq.print_quant_summary(encoder_int8)

    # print("\n=== Quantizing message GNN with SmoothQuant INT8 ===")
    # msg_forward_loop = make_forward_loop_message(msg_loader, dev)
    # message_int8 = mtq.quantize(message, config, msg_forward_loop)
    # mtq.print_quant_summary(message_int8)

    # print("\n=== Quantizing bev GNN with SmoothQuant INT8 ===")
    # bev_forward_loop = make_forward_loop_bev(bev_loader, dev)
    # bev_int8 = mtq.quantize(bev, config, bev_forward_loop)
    # mtq.print_quant_summary(bev_int8)

    # print("\n=== Quantizing bev decoder with SmoothQuant INT8 ===")
    # bev_dec_forward_loop = make_forward_loop_bev_dec(bev_dec_loader, dev)
    # bev_dec_int8 = mtq.quantize(bev_dec, config, bev_dec_forward_loop)
    # mtq.print_quant_summary(bev_dec_int8)

    # --------------------
    # Example inputs for ONNX export (match your earlier export script)
    # --------------------
    # encoder: [B,3,H,W]
    enc_example = torch.randn(1, 3, resolution, resolution, device=dev)

    # message / bev: [B, seq_len, in_channels] etc.
    x_i_example = torch.randn(1, gnn_in_seq_len, gnn_in_channels, device=dev)
    x_j_example = torch.randn(1, gnn_in_seq_len, gnn_in_channels, device=dev)
    edge_pred_example = torch.randn(1, 17, device=dev)

    # bev output features
    bev_features_example = torch.randn(1, gnn_out_channels, device=dev)

    # --------------------
    # Export quantized modules to ONNX
    # --------------------
    print("\n=== Exporting SmoothQuant INT8 ONNX models ===")

    # Encoder
    export_to_onnx(
        encoder_int8,
        enc_example,
        out_dir / "0kc5po4ee18_int8_smoothquant_onnx_cuda_enc_trimmed.onnx",
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )

    # Message GNN
    # export_to_onnx(
    #     message_int8,
    #     (x_i_example, x_j_example),
    #     out_dir / "0kc5po4ee18_int8_smoothquant_onnx_cuda_msg.onnx",
    #     input_names=["features_i", "features_j"],
    #     output_names=["edge_predictions"],
    #     dynamic_axes={
    #         "features_i": {0: "batch_size"},
    #         "features_j": {0: "batch_size"},
    #         "edge_predictions": {0: "batch_size"},
    #     },
    # )

    # # Bev GNN
    # export_to_onnx(
    #     bev_int8,
    #     (x_i_example, x_j_example, edge_pred_example),
    #     out_dir / "0kc5po4ee18_int8_smoothquant_onnx_cuda_bev.onnx",
    #     input_names=["features_i", "features_j", "edge_prediction"],
    #     output_names=["bev_features"],
    #     dynamic_axes={
    #         "features_i": {0: "batch_size"},
    #         "features_j": {0: "batch_size"},
    #         "edge_prediction": {0: "batch_size"},
    #         "bev_features": {0: "batch_size"},
    #     },
    # )

    # # Bev decoder
    # export_to_onnx(
    #     bev_dec_int8,
    #     bev_features_example,
    #     out_dir / "0kc5po4ee18_int8_smoothquant_onnx_cuda_bevdec.onnx",
    #     input_names=["bev_features"],
    #     output_names=["bev_map"],
    #     dynamic_axes={
    #         "bev_features": {0: "batch_size"},
    #         "bev_map": {0: "batch_size"},
    #     },
    # )

    print("\n✅ SmoothQuant + ONNX export completed.")


if __name__ == "__main__":
    main()
