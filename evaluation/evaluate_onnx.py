import argparse
import glob
from pathlib import Path
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch_geometric
from torch_geometric.utils import dropout_edge, add_self_loops
from torch.utils.dlpack import to_dlpack, from_dlpack
from .utils import radius_graph

# onnxruntime is optional; we handle "not installed" gracefully
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# Your utils
from evaluation.utils import evaluate_model_accruacy, analyze_model_runtime  # noqa: F401

# ----------------------------
# Helper: ONNX session wrapper
# ----------------------------

class OrtModule:
    """
    Minimal callable wrapper that runs ONNX Runtime on CUDA using DLPack
    (no host copies). Assumes session was created with a CUDA (or TensorRT) EP.
    """
    def __init__(self, session: "ort.InferenceSession", device_id: int = 0, force_fp16: bool = True):
        self.session = session
        self.device_id = device_id
        self.force_fp16 = force_fp16
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

        # Basic sanity: make sure a GPU EP is available
        eps = [p for p in self.session.get_providers()]
        if not any(ep in eps for ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider")):
            raise RuntimeError(f"Session has no GPU EP. Providers={eps}")

    def __call__(self, *inputs: torch.Tensor):
        feeds = {}
        for name, t in zip(self._input_names, inputs):
            if self.force_fp16 and t.dtype != torch.float16:
                t = t.to(torch.float16)
            if not t.is_cuda:
                t = t.cuda(self.device_id)
            t = t.contiguous()

            # Zero-copy handoff to ORT via DLPack (stays on GPU)
            feeds[name] = ort.OrtValue.from_dlpack(to_dlpack(t))

        # Run and get OrtValue outputs (on GPU)
        ort_outs = self.session.run_with_ort_values(self._output_names, feeds, run_options=None)

        # Convert back to torch Tensors on GPU (zero-copy)
        torch_outs = [from_dlpack(o.to_dlpack()) for o in ort_outs]
        return torch_outs[0] if len(torch_outs) == 1 else tuple(torch_outs)

# ------------------------------------------
# Helpers: path discovery & safe torch.load
# ------------------------------------------
def _first_or_none(patterns):
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            return Path(matches[0])
    return None

def find_ts_paths(ts_dir: Path):
    """
    Try to locate the five component TorchScript files under ts_dir.
    Falls back to loose globs if the exact pattern isn't found.
    """
    enc = _first_or_none([str(ts_dir / "*_jit_*_enc.ts"), str(ts_dir / "*enc.ts")])
    msg = _first_or_none([str(ts_dir / "*_jit_*_msg.ts"), str(ts_dir / "*msg.ts")])
    post = _first_or_none([str(ts_dir / "*_jit_*_post.ts"), str(ts_dir / "*post.ts")])
    bev = _first_or_none([str(ts_dir / "*_jit_*_bev.ts"), str(ts_dir / "*bev.ts")])
    bev_dec = _first_or_none([str(ts_dir / "*_jit_*_bevdec.ts"), str(ts_dir / "*bevdec.ts")])

    return dict(enc=enc, msg=msg, post=post, bev=bev, bev_dec=bev_dec)

def find_onnx_paths(onnx_dir: Path):
    enc = Path("0kc5po4ee18_int8_max_modelopt_from_onnx_fp16_cuda_enc.onnx")
    msg = Path("0kc5po4ee18_int8_max_modelopt_from_onnx_fp16_cuda_msg.onnx")
    post = Path("0kc5po4ee18_float32_jit_cpu_post.ts")
    bev = Path("0kc5po4ee18_int8_max_modelopt_from_onnx_fp16_cuda_bev.onnx")
    bev_dec = Path("0kc5po4ee18_int8_max_modelopt_from_onnx_fp16_cuda_bevdec.onnx")

    return dict(enc=enc, msg=msg, post=post, bev=bev, bev_dec=bev_dec)

# ----------------------------
# Loaders
# ----------------------------
def load_ts_models(paths: dict, device: torch.device, dtype: torch.dtype):
    models = {}
    for k, p in paths.items():
        if p is None:
            raise FileNotFoundError(f"Missing TorchScript file for: {k}")
        m = torch.jit.load(str(p), map_location=device).eval()
        # Keep PosePost (post) in float32 for roma stability if that’s how you exported it
        if k == "post":
            m = m.to(torch.float32)
        else:
            m = m.to(dtype)
        models[k] = m
    return models

class TSModule:
    """Light wrapper to behave like a callable module (similar to OrtModule)."""
    def __init__(self, ts_mod: torch.jit.ScriptModule, force_fp32_inputs: bool = True):
        self.mod = ts_mod.eval()
        self.force_fp32_inputs = force_fp32_inputs

    def to(self, *args, **kwargs):
        self.mod.to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        # Optionally upcast inputs to FP32 for numerical stability in PosePost.
        if self.force_fp32_inputs:
            def _cast(x):
                if isinstance(x, torch.Tensor):
                    return x.float()
                return x
            args   = tuple(_cast(a) for a in args)
            kwargs = {k: _cast(v) for k, v in kwargs.items()}
        return self.mod(*args, **kwargs)

def load_onnx_modules(paths: dict):
    device = torch.device("cuda")
                          
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime is not installed. Install it to benchmark ONNX models.")
    mods = {}
    for k, p in paths.items():
        if p is None:
            continue
        if k == "post":
            ts_mod = torch.jit.load(str(p), map_location=device)
            ts_mod = ts_mod.to(device).eval()
            mods[k] = TSModule(ts_mod, force_fp32_inputs=True)  # PosePost prefers FP3

        else:
            sess = ort.InferenceSession(
                str(p),
                providers=[
                    ("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": True}),
                    ("CUDAExecutionProvider",     {"device_id": 0}),
                    "CPUExecutionProvider",
                ],
            )
            mods[k] = OrtModule(sess)
    return mods

# ------------------------------------------
# Synthetic data & staged feature generation
# ------------------------------------------
@torch.inference_mode()
def make_dummy_batch(device, resolution=224, bs=1, n_agents=3):
    return {
        "pos": torch.rand(bs, n_agents, 3, device=device) * 8,
        "rot_quat": torch.rand(bs, n_agents, 4, device=device),
        "img_norm": torch.randn(bs, n_agents, 3, resolution, resolution, device=device) * 3,
    }

@torch.inference_mode()
def stage_features_with_encoder(encoder_callable, img_batch, dtype, device):
    """
    Run encoder twice to get x_i, x_j for a pair of agents (0,1).
    Returns (x_i, x_j) on device with given dtype.
    """
    img_flat = img_batch.flatten(0, 1).to(dtype)
    img_i = img_flat[0].unsqueeze(0)
    img_j = img_flat[1].unsqueeze(0)
    # TS: returns torch tensors; ORT: OrtModule returns tensor as well
    x_i = encoder_callable(img_i.to(device))
    x_j = encoder_callable(img_j.to(device))
    # If ORT produced CPU tensors, move to device
    if isinstance(x_i, torch.Tensor):
        x_i = x_i.to(device, dtype=dtype)
    if isinstance(x_j, torch.Tensor):
        x_j = x_j.to(device, dtype=dtype)
    return (img_i, img_j), (x_i, x_j)

@torch.inference_mode()
def stage_msg(message_callable, x_i, x_j, dtype, device):
    aggr = message_callable(x_i.to(device, dtype=dtype), x_j.to(device, dtype=dtype))
    # ORT may return CPU
    aggr = aggr if aggr.device.type == device.type else aggr.to(device)
    return aggr

@torch.inference_mode()
def stage_bev(bev_callable, x_i, x_j, aggr, dtype, device):
    bev = bev_callable(
        x_i.to(device, dtype=dtype), x_j.to(device, dtype=dtype), aggr.to(device, dtype=dtype)
    )
    bev = bev if bev.device.type == device.type else bev.to(device)
    return bev

# --- full-model wrapper for eval ---
@torch.no_grad()
class ONNXEnsembleModule(torch.nn.Module):
    """
    Uses ONNXRuntime-backed submodules in `onnx_mods` to emulate:
      BEVGNNModel.forward(...)
    Returns:
      ((edge_preds_proc_dict, bev_nodes), (edge_index_pose, graphs_batch))
    """
    def __init__(self, onnx_mods: dict, comm_range: float = 1000.0):
        super().__init__()
        self.onnx_mods = onnx_mods
        # default if caller doesn't pass one; matches original semantics
        self.comm_range = comm_range

    @torch.no_grad()
    def forward(self, input):
        device = input["img_norm"].device
        dtype = torch.float16

        # ---- batch + flatten ----
        bs, n_nodes = input["img_norm"].shape[:2]
        graphs_batch = torch.repeat_interleave(
            torch.arange(bs, device=device), n_nodes, dim=0
        )

        # ---- Encoder(+post) -> node token sequences (fp16) ----
        img_flat = input["img_norm"].flatten(0, 1).to(dtype)               # [B*N,3,H,W]
        x = self.onnx_mods["enc"](img_flat)                                # [B*N, S, C] fp16
        x = x.to(dtype)

        # ---- Build pose graph (radius over positions) ----
        pos = input["pos"].flatten(0, 1)                                   # [B*N, 3]
        edge_index_pose = radius_graph(
            pos, r=self.comm_range, batch=graphs_batch, loop=False
        )                                                                   # [2, E]

        # ---- Edge message prediction: (target <- source) ----
        # Align with PyG convention: edge_index[0]=source, edge_index[1]=target.
        src, dst = edge_index_pose[0], edge_index_pose[1]
        import pdb; pdb.set_trace()
        x_i_msg = x[dst]                                                   # [E, S, C] target
        x_j_msg = x[src]                                                   # [E, S, C] source
        edge_preds = self.onnx_mods["msg"](x_i_msg.to(dtype), x_j_msg.to(dtype)).to(dtype)  # [E,17]

        # ---- Pose post-processing (rot as unit quat, vars exp) ----
        # onnx "post" returns tuple: (pos, pos_var, rot, rot_var)
        pos_p, pos_var_p, rot_p, rot_var_p = self.onnx_mods["post"](edge_preds)
        # Match original dict (rot_var [E,1])
        edge_preds_proc = {
            "pos":       pos_p.to(dtype),
            "pos_var":   pos_var_p.to(dtype),
            "rot":       rot_p.to(dtype),
            "rot_var":   rot_var_p.unsqueeze(-1).to(dtype),
        }

        # ---- Edge dropout, then add self-loops with zero edge_attr ----
        edge_index_drop, edge_mask = torch_geometric.utils.dropout_edge(
            edge_index_pose, p=0.3, training=False
        )
        edge_attr_drop = edge_preds[edge_mask]                              # [E',17] fp16
        edge_index_self, edge_attr_self = torch_geometric.utils.add_self_loops(
            edge_index_drop, edge_attr_drop, fill_value=0.0
        )                                                                   # [2, E_s], [E_s,17] (fp16)

        # ---- BEV edge aggregation with ONNX Bev, scatter->node ----
        src_b, dst_b = edge_index_self[0], edge_index_self[1]
        x_i_bev = x[dst_b]                                                  # target node features
        x_j_bev = x[src_b]                                                  # source node features
        bev_edge_vec = self.onnx_mods["bev"](
            x_i_bev.to(dtype), x_j_bev.to(dtype), edge_attr_self.to(dtype)
        ).to(dtype)                                                         # [E_s, C_bev]

        # scatter-mean into target nodes
        num_nodes = bs * n_nodes
        C_bev = bev_edge_vec.shape[-1]
        bev_sum = torch.zeros((num_nodes, C_bev), device=device, dtype=dtype)
        bev_cnt = torch.bincount(dst_b, minlength=num_nodes).clamp_min(1).unsqueeze(1).to(bev_sum.dtype)
        bev_sum.index_add_(0, dst_b, bev_edge_vec)
        bev_node = bev_sum / bev_cnt                                        # [B*N, C_bev] fp16

        # ---- BEV decoder -> cropped map (1,60,60), reshape back to [B,N,...] ----
        bev_maps = self.onnx_mods["bev_dec"](bev_node.to(dtype))            # [B*N, 1,60,60] fp16
        bev_nodes = bev_maps.view(bs, n_nodes, *bev_maps.shape[1:]).to(dtype)

        return (edge_preds_proc, bev_nodes), (edge_index_pose, graphs_batch)



# ------------------------------------------
# Benchmark routines (TS and ONNX)
# ------------------------------------------
@torch.inference_mode()
def benchmark_ts(ts_models: dict, data: dict, dtype: torch.dtype, device: torch.device, dataset=None):
    print("\n=== TorchScript pipeline ===")
    img = data["img_norm"].to(device)

    # 1) Encoder
    (img_i, img_j), (x_i, x_j) = stage_features_with_encoder(ts_models["enc"], img, dtype, device)
    analyze_model_runtime(ts_models["enc"], img_i, name="TS Encoder_i")
    analyze_model_runtime(ts_models["enc"], img_j, name="TS Encoder_j")

    # 2) Message (edge)
    analyze_model_runtime(ts_models["msg"], x_i, x_j, name="TS Msg Module")
    aggr = stage_msg(ts_models["msg"], x_i, x_j, dtype, device)

    # 3) PosePost (often kept in fp32 for roma)
    # Note: just time it as well; returns tuple (pos, pos_unc, rot, rot_unc)
    analyze_model_runtime(ts_models["post"], aggr.to(torch.float32), name="TS PosePost")

    # 4) BEV GNN
    analyze_model_runtime(ts_models["bev"], x_i, x_j, aggr, name="TS BEV Module")
    bev_feats = stage_bev(ts_models["bev"], x_i, x_j, aggr, dtype, device)

    # 5) Decoder
    analyze_model_runtime(ts_models["bev_dec"], bev_feats, name="TS BEV Decoder")


@torch.inference_mode()
def benchmark_onnx(onnx_mods: dict, data: dict, dtype: torch.dtype, device: torch.device, dataset=None):
    print("\n=== ONNX (onnxruntime) pipeline ===")
    if not ORT_AVAILABLE:
        print("onnxruntime not available; skipping ONNX benchmark.")
        return

    img = data["img_norm"]  # ORT wrapper expects CPU numpy, we’ll pass tensors and wrapper will convert

    # 1) Encoder
    (img_i, img_j), (x_i, x_j) = stage_features_with_encoder(onnx_mods["enc"], img, dtype, torch.device("cpu"))
    # evaluate_model_runtime: pass the OrtModule and the *inputs* (it converts internally)
    analyze_model_runtime(onnx_mods["enc"], img_i, name="ONNX Encoder_i")
    analyze_model_runtime(onnx_mods["enc"], img_j, name="ONNX Encoder_j")

    # Make sure subsequent inputs live where OrtModule expects (CPU tensors ok)
    x_i_cpu = x_i.to("cpu")
    x_j_cpu = x_j.to("cpu")

    # 2) Message
    analyze_model_runtime(onnx_mods["msg"], x_i_cpu, x_j_cpu, name="ONNX Msg Module")
    aggr = onnx_mods["msg"](x_i_cpu, x_j_cpu)  # CPU tensor

    # 3) PosePost (multiple outputs); we can still time it
    #analyze_model_runtime(onnx_mods["post"], aggr.to("cpu", dtype=torch.float32), name="ONNX PosePost")

    # 4) BEV
    analyze_model_runtime(onnx_mods["bev"], x_i_cpu, x_j_cpu, aggr, name="ONNX BEV Module")
    bev_feats = onnx_mods["bev"](x_i_cpu, x_j_cpu, aggr)

    # 5) Decoder
    analyze_model_runtime(onnx_mods["bev_dec"], bev_feats, name="ONNX BEV Decoder")
    

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load existing TorchScript & ONNX component models and benchmark both pipelines."
    )
    parser.add_argument("--ts_dir", type=str, default="./models_exported",
                        help="Directory containing *.ts components.")
    parser.add_argument("--onnx_dir", type=str, default="./models_exported",
                        help="Directory containing *.onnx components.")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_agents", type=int, default=3, help="Must be >= 2.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float16

    ts_dir = Path(args.ts_dir)
    onnx_dir = Path(args.onnx_dir)

    # --- load paths
    ts_paths = find_ts_paths(ts_dir)
    onnx_paths = find_onnx_paths(onnx_dir)

    print("Resolved TorchScript component paths:")
    for k, v in ts_paths.items():
        print(f"  {k}: {v if v else 'NOT FOUND'}")
    print("\nResolved ONNX component paths:")
    for k, v in onnx_paths.items():
        print(f"  {k}: {v if v else 'NOT FOUND'}")

    # --- load models
    # ts_models = load_ts_models(ts_paths, device=device, dtype=dtype)

    onnx_mods = load_onnx_modules(onnx_paths)

    # --- fabricate a small batch and run both pipelines
    data = make_dummy_batch(device=device, resolution=args.resolution,
                            bs=args.batch_size, n_agents=max(2, args.n_agents))

    # Evaluate runtime
    # benchmark_ts(ts_models, data, dtype=dtype, device=device)
    # benchmark_onnx(onnx_mods, data, dtype=dtype, device=device)

    # Evaluate performance
    from train.dataloader import RelPosDataModule
    import yaml
    with open('/mnt/beegfs/lchang2/CoViS-Net/train/configs/covisnet.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    dataloader = RelPosDataModule(**config["data"])
    dataloader.setup(stage=None)
 
    test_loader = dataloader.val_dataloader()

    # For perf timing you can still grab one batch:
    # benchmark_ts(ts_models, data, dtype=dtype, device=device, dataset=test_loader)
    # benchmark_onnx(onnx_mods, data, dtype=dtype, device=device, dataset=test_loader)

    model = ONNXEnsembleModule(onnx_mods)
    results = evaluate_model_accruacy(model, test_loader)
    print("!!!!", results)

if __name__ == "__main__":
    main()
