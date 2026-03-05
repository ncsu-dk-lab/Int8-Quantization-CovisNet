import os
import torch
import torch.nn as nn
import tensorrt as trt
from .utils import radius_graph
from torch_geometric.utils import dropout_edge
import torch_geometric
import roma
from evaluation.utils import evaluate_model_accruacy




from torch.utils.data import IterableDataset, DataLoader
# ---------- Generic TensorRT engine runner (FP16-friendly) ----------

import os
import torch
import tensorrt as trt

class TRTEngine:
    def __init__(self, engine_path: str):
        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
        
        # Initialize TRT logger and runtime
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(self.logger, "") # Good practice if your model uses TRT plugins

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

        self.context: trt.IExecutionContext = self.engine.create_execution_context()

        # Cache I/O tensor names (TRT-10 tensor API)
        n_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(n_io)]
        self.input_names  = [n for n in self.tensor_names 
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names 
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __del__(self):
        """Safely clean up C++ TensorRT bindings to prevent GPU memory leaks."""
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine

    # --- utils ---
    def _torch_dtype_from_trt(self, t: trt.DataType) -> torch.dtype:
        if t == trt.DataType.HALF:  return torch.float16
        if t == trt.DataType.FLOAT: return torch.float32
        if t == trt.DataType.INT32: return torch.int32
        if t == trt.DataType.INT8:  return torch.int8
        if t == trt.DataType.BOOL:  return torch.bool
        # Added INT64 as it is common for indices/argmax outputs
        if t == trt.DataType.INT64: return torch.int64 
        raise ValueError(f"Unsupported TRT dtype: {t}")

    def __call__(self, profile_index: int = 0, **tensors):
        """
        Run inference on the TensorRT engine using the Tensor API (TRT 10).
        """
        if not tensors:
            raise ValueError("No input tensors provided to TRTEngine.__call__")

        # --- A) Derive device & CUDA stream from the first input tensor ---
        first_in = next(iter(tensors.values()))
        if not first_in.is_cuda:
            raise AssertionError(f"All inputs must be CUDA tensors; got {first_in.device}")
        
        device = first_in.device
        stream = torch.cuda.current_stream(device).cuda_stream

        # --- 1) Get profile bounds, validate input shapes & set shapes ---
        for name in self.input_names:
            if name not in tensors:
                raise KeyError(
                    f"Missing required input: '{name}'. Available inputs: {self.input_names}"
                )

            t = tensors[name]
            if not t.is_cuda:
                raise AssertionError(f"Input '{name}' must be CUDA, got device={t.device}")

            # Cast FP32->FP16 if engine expects HALF and user gave FLOAT
            expected_dtype = self.engine.get_tensor_dtype(name)
            if expected_dtype == trt.DataType.HALF and t.dtype == torch.float32:
                t = t.to(torch.float16)
                tensors[name] = t

            # Retrieve profile bounds for dynamic shapes
            mn, opt, mx = self.engine.get_tensor_profile_shape(name, profile_index)
            shp = tuple(int(x) for x in t.shape)

            # Rank / bounds checks
            if len(shp) != len(mx):
                raise AssertionError(
                    f"Rank mismatch for '{name}': got {shp}, engine rank {len(mx)}"
                )
            for d, lo, hi in zip(shp, mn, mx):
                if not (lo <= d <= hi):
                    raise AssertionError(
                        f"Shape {shp} for '{name}' violates profile [{mn}..{mx}]"
                    )

            # Set input shape on the context using TRT 10 API
            if not self.context.set_input_shape(name, shp):
                raise RuntimeError(f"set_input_shape failed for '{name}' with shape {shp}")

        # --- 2) Sanity check: shapes must now be concrete (no -1) ---
        for name in self.input_names:
            s = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in s):
                raise RuntimeError(f"After set_input_shape, '{name}' is still dynamic: {s}.")

        # --- 3) Allocate outputs from context's tensor shapes ---
        outputs = {}
        batch = int(next(iter(tensors.values())).shape[0]) # Fallback for unresolved dimensions

        for name in self.output_names:
            s = list(self.context.get_tensor_shape(name))
            for i, d in enumerate(s):
                if d < 0:      
                    s[i] = batch  # Resolve dynamic batch dim
            if any(d <= 0 for d in s):
                raise RuntimeError(f"Unresolved output shape for '{name}': {s}")
            
            out_dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(tuple(s), dtype=out_dtype, device=device)

        # --- 4) Register tensor addresses (inputs + outputs) ---
        for name in self.input_names:
            if not self.context.set_tensor_address(name, tensors[name].data_ptr()):
                raise RuntimeError(f"Failed to set address for input '{name}'")

        for name in self.output_names:
            if not self.context.set_tensor_address(name, outputs[name].data_ptr()):
                raise RuntimeError(f"Failed to set address for output '{name}'")

        # --- 5) Execute on the chosen CUDA stream ---
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 failed. Check input/output shapes.")

        # Sync PyTorch with TRT stream to make sure outputs are ready
        torch.cuda.current_stream(device).synchronize()

        # --- 6) Adjust output views to runtime shapes (for data-dependent shapes) ---
        for name in self.output_names:
            rt_shape = tuple(self.context.get_tensor_shape(name))
            if all(d >= 0 for d in rt_shape) and rt_shape != tuple(outputs[name].shape):
                outputs[name] = outputs[name].view(rt_shape)

        return outputs

# ---------- The assembled FP16 model using your engines ----------
class AssembledBEVGNNTRT(nn.Module):
    def __init__(
        self,
        comm_range: float = 1000.0,
        gnn_in_channels: int = 24,     # not used inside engines but kept for parity
        gnn_in_seq_len: int = 128,      # expected 128
        bev_gnn_out_channels: int = 384, # expected 384
        dec_out_channels: int = 1,    # expected 1
        engine_dir: str = ".",
        enc_engine: str = "enc_int8_SmoothQ_512.engine",
        msg_engine: str = "msg_int8_SmoothQ_512.engine",
        bev_engine: str = "bev_int8_SmoothQ_512.engine",
        bevdec_engine: str = "bevdec_int8_SmoothQ_512.engine",
        posepost_ts: str = "models/0kc5po4ee18_float32_jit_cuda_post.ts",  # TorchScript (TRT-compiled)
        dtype=torch.float32,
    ):
        super().__init__()
        self.comm_range = comm_range
        self.gnn_in_seq_len = gnn_in_seq_len
        self.bev_gnn_out_channels = bev_gnn_out_channels

        # Load TRT engines
        self.enc_trt     = TRTEngine(os.path.join(engine_dir, enc_engine))     # input: image -> output: features [N, 128, 24]
        self.msg_trt     = TRTEngine(os.path.join(engine_dir, msg_engine))     # inputs: features_i, features_j -> edge_pred [N_e, 17]
        self.bev_trt     = TRTEngine(os.path.join(engine_dir, bev_engine))     # inputs: features_i, features_j, edge_prediction -> [N_e, 384]
        self.bevdec_trt  = TRTEngine(os.path.join(engine_dir, bevdec_engine))  # input: bev_features [N_nodes, 384] -> [N_nodes, 1, 60, 60]

        # PosePost (compiled to TorchScript by Torch-TensorRT)
        assert os.path.exists(os.path.join(engine_dir, posepost_ts)), f"Missing {posepost_ts}"
        self.pose_post = torch.jit.load(os.path.join(engine_dir, posepost_ts)).eval().to("cuda")
        self.dtype = dtype

    
    @torch.no_grad()
    def forward(self, input):
        bs, n_nodes = input["img_norm"].shape[:2]

        img_flat = input["img_norm"].flatten(0, 1).to(self.dtype).contiguous()

        # 1) Run encoder engine: pass tensor as **named** input
        enc_out_dict = self.enc_trt(
            profile_index=0,             # optional, defaults to 0 anyway
            image=img_flat,           # <-- USE THE REAL INPUT NAME HERE
        )

        # 2) Get the encoder features (assuming single output)
        enc_feats = enc_out_dict[self.enc_trt.output_names[0]]

        # 3) Build graphs batch
        graphs = torch_geometric.data.Batch()
        graphs.x = enc_feats
        graphs.batch = torch.repeat_interleave(
            torch.arange(bs, device=input["img_norm"].device),
            n_nodes,
            dim=0,
        )
        graphs.pos = input["pos"].flatten(0, 1)
        graphs.rot = input["rot_quat"].flatten(0, 1)

        edge_index_pose = radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )
        graphs.edge_index = edge_index_pose

        x = graphs.x
        edge_index = graphs.edge_index

        edge_preds = self.msg_trt(
            features_i=x[edge_index[1]],
            features_j=x[edge_index[0]],
        )[self.msg_trt.output_names[0]]

        edge_preds_proc = {
            "pos": edge_preds[:, 0:3],
            "pos_var": edge_preds[:, 3:6].exp(),
            "rot": roma.symmatrixvec_to_unitquat(
                edge_preds[:, 6:16].to(torch.float)
            ).to(edge_preds.dtype),
            "rot_var": edge_preds[:, 16:17].exp(),
        }

        graphs.edge_index, edge_mask = dropout_edge(graphs.edge_index, p=0.3)
        graphs.edge_attr = edge_preds[edge_mask]

        return (edge_preds_proc, None), (edge_index_pose, graphs.batch)
    
def make_dummy_batch(device="cuda", resolution=224, bs=1, n_agents=3):
    return {
        "pos": torch.rand(bs, n_agents, 3, device=device) * 8,
        "rot_quat": torch.rand(bs, n_agents, 4, device=device),
        "img_norm": torch.randn(bs, n_agents, 3, resolution, resolution, device=device) * 3,
    }

class DummyBatchStream(IterableDataset):
    def __init__(self, device="cuda", resolution=224, bs=1, n_agents=3, num_batches=32):
        self.device = device
        self.resolution = resolution
        self.bs = bs
        self.n_agents = n_agents
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield make_dummy_batch(
                device=self.device,
                resolution=self.resolution,
                bs=self.bs,
                n_agents=self.n_agents,
            )

def main():
    from train.dataloader import RelPosDataModule
    import yaml
    with open('/mnt/beegfs/lchang2/CoViS-Net/train/configs/covisnet.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    dataloader = RelPosDataModule(**config["data"])
    dataloader.setup(stage=None)
    test_loader = dataloader.val_dataloader()

    print("Starting evaluation...")
    model = AssembledBEVGNNTRT()
    results = evaluate_model_accruacy(model, test_loader)
    print("!!!!", results)
    
    # print("Starting to load TRT engine...")
    # engine = TRTEngine("enc_int8_SmoothQ.engine")
    # x = torch.randn(96, 3, 224, 224, device="cuda", dtype=torch.float32)
    # with torch.no_grad():
    #     out = engine(image=x)  # profile_index=0 by default

    # print(engine.input_names, engine.output_names)
    # print({k: v.shape for k, v in out.items()})

    # FP16: Dist_cm: 45.5711, Angle_deg: 7.2290
    # FP32: Dist_cm: 45.6932, Angle_deg: 7.1933
    # Int8_SQ: Dist_cm: 45.1763, Angle_deg: 7.2183
    # Int8_SQ_512: Dist_cm: 46.6567, Angle_deg: 7.0947
    # Int8_SQ_200: Dist_cm: 47.1447, Angle_deg: 7.2265
    # Int8_etp_200: Dist_cm: 103.0701, Angle_deg: 135.2298
if __name__ == "__main__":
    main()