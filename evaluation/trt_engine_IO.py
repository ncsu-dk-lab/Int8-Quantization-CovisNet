import os
import torch
import torch.nn as nn
import tensorrt as trt
from .utils import radius_graph
from torch_geometric.utils import dropout_edge
import torch_geometric
import roma


from torch.utils.data import IterableDataset, DataLoader
# ---------- Generic TensorRT engine runner (FP16-friendly) ----------

class TRTEngine:
    def __init__(self, engine_path: str):
        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
        logger = trt.Logger(trt.Logger.VERBOSE)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())

        self.context: trt.IExecutionContext = self.engine.create_execution_context()

        # Torch CUDA stream handle (int) for execute_async_v3
        self.stream = torch.cuda.current_stream().cuda_stream

        # Cache I/O tensor names (TRT-10 tensor API)
        n_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(n_io)]
        self.input_names  = [n for n in self.tensor_names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    # --- utils ---
    def _torch_dtype_from_trt(self, t: trt.DataType):
        if t == trt.DataType.HALF:  return torch.float16
        if t == trt.DataType.FLOAT: return torch.float32
        if t == trt.DataType.INT32: return torch.int32
        if t == trt.DataType.INT8:  return torch.int8
        if t == trt.DataType.BOOL:  return torch.bool
        raise ValueError(f"Unsupported TRT dtype: {t}")

    def __call__(self, profile_index: int = 0, **tensors):
        """
        Run inference on the TensorRT engine using the Tensor API (TRT 8+/10).

        Usage:
            outputs = engine(image=img_tensor)
            # where "image" matches the real TRT input tensor name
        """
        if not tensors:
            raise ValueError("No input tensors provided to TRTEngine.__call__")

        # --- A) Derive device & CUDA stream from the first input tensor ---
        first_in = next(iter(tensors.values()))
        if not first_in.is_cuda:
            raise AssertionError(f"All inputs must be CUDA tensors; got {first_in.device}")
        device = first_in.device
        # Use the *current* PyTorch stream on that device
        stream = torch.cuda.current_stream(device).cuda_stream

        # --- 0) Select optimization profile BEFORE shapes/addresses ---
        set_prof_async = getattr(self.context, "set_optimization_profile_async", None)
        if set_prof_async is not None:
            ok = set_prof_async(profile_index, stream)
            if not ok:
                raise RuntimeError(f"set_optimization_profile_async({profile_index}) failed")
        else:
            set_prof = getattr(self.context, "set_optimization_profile", None)
            if set_prof is None:
                raise RuntimeError("ExecutionContext has no optimization profile API")
            ok = set_prof(profile_index)
            if not ok:
                raise RuntimeError(f"set_optimization_profile({profile_index}) failed")

        # --- 1) Get profile bounds, validate input shapes & set shapes ---
        profile_bounds = {}
        for name in self.input_names:
            mn, opt, mx = self.engine.get_tensor_profile_shape(name, profile_index)
            profile_bounds[name] = (tuple(mn), tuple(opt), tuple(mx))
            # You can uncomment for debugging:
            # print(f"[TRT] {name}: min={mn}, opt={opt}, max={mx}")

        # Validate inputs and set shapes in the context
        for name in self.input_names:
            if name not in tensors:
                raise KeyError(
                    f"Missing required input: '{name}'. "
                    f"Available inputs: {self.input_names}"
                )

            t = tensors[name]
            if not t.is_cuda:
                raise AssertionError(f"Input '{name}' must be CUDA, got device={t.device}")

            # Cast FP32->FP16 if engine expects HALF and user gave FLOAT
            if self.engine.get_tensor_dtype(name) == trt.DataType.HALF and t.dtype == torch.float32:
                t = t.to(torch.float16)
                tensors[name] = t

            shp = tuple(int(x) for x in t.shape)
            mn, _, mx = profile_bounds[name]

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

            # Set input shape on the context (tensor API)
            if hasattr(self.context, "set_input_shape"):
                ok = self.context.set_input_shape(name, shp)
            else:
                ok = self.context.set_tensor_shape(name, shp)
            if not ok:
                raise RuntimeError(f"set_input_shape failed for '{name}' with shape {shp}")

        # --- 2) Sanity check: shapes must now be concrete (no -1) ---
        for name in self.input_names:
            s = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in s):
                raise RuntimeError(
                    f"After set_input_shape, '{name}' is still dynamic: {s}. "
                    f"Check that you used the correct tensor names / API."
                )

        # --- 3) Allocate outputs from context's tensor shapes ---
        # Use batch dimension from the first input as a fallback for dynamic -1s
        batch = int(next(iter(tensors.values())).shape[0])

        def _alloc_out_shape(name: str):
            s = list(self.context.get_tensor_shape(name))
            for i, d in enumerate(s):
                if d < 0:      # e.g. dynamic batch dimension
                    s[i] = batch
            if any(d <= 0 for d in s):
                raise RuntimeError(f"Unresolved output shape for '{name}': {s}")
            return tuple(s)

        outputs = {}
        for name in self.output_names:
            out_dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(
                _alloc_out_shape(name),
                dtype=out_dtype,
                device=device,
            )

        # --- 4) Register tensor addresses (inputs + outputs) ---
        # Prefer set_input_tensor_address if available; fall back to set_tensor_address.
        set_input_addr = getattr(self.context, "set_input_tensor_address", None)
        if set_input_addr is None:
            set_input_addr = self.context.set_tensor_address

        # Inputs
        for name in self.input_names:
            ptr = tensors[name].data_ptr()
            ok = set_input_addr(name, ptr)
            if not ok:
                raise RuntimeError(f"Failed to set address for input '{name}'")

        # Outputs
        for name in self.output_names:
            ptr = outputs[name].data_ptr()
            ok = self.context.set_tensor_address(name, ptr)
            if not ok:
                raise RuntimeError(f"Failed to set address for output '{name}'")

        # --- 5) Final guard: shapes must still be concrete before enqueue ---
        for name in self.input_names:
            s = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in s):
                raise RuntimeError(
                    f"Input '{name}' still unresolved before enqueue: {s}"
                )

        # --- 6) Execute on the chosen CUDA stream ---
        ok = self.context.execute_async_v3(stream)
        if not ok:
            dbg_inputs = {n: tuple(self.context.get_tensor_shape(n)) for n in self.input_names}
            dbg_outs   = {n: tuple(self.context.get_tensor_shape(n)) for n in self.output_names}
            raise RuntimeError(
                "TensorRT execute_async_v3 failed.\n"
                f"Resolved input shapes: {dbg_inputs}\n"
                f"Resolved output shapes: {dbg_outs}\n"
            )

        # Optional: sync PyTorch with TRT stream to make sure outputs are ready
        torch.cuda.current_stream(device).synchronize()

        # --- 7) Adjust output views to runtime shapes, if they changed ---
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
        enc_engine: str = "enc_int8_SmoothQ.engine",
        msg_engine: str = "msg_int8_SmoothQ.engine",
        bev_engine: str = "bev_int8_SmoothQ.engine",
        bevdec_engine: str = "bevdec_int8_SmoothQ.engine",
        posepost_ts: str = "models/0kc5po4ee18_float32_trt_post.ts",  # TorchScript (TRT-compiled)
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
    # from train.dataloader import RelPosDataModule
    # import yaml
    # with open('/mnt/beegfs/lchang2/CoViS-Net/train/configs/covisnet.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
        
    # dataloader = RelPosDataModule(**config["data"])
    # dataloader.setup(stage=None)
    # test_loader = dataloader.val_dataloader()

    # model = AssembledBEVGNNTRT()
    # results = evaluate_model_accruacy(model, test_loader)
    # print("!!!!", results)
    
    print("Starting to load TRT engine...")
    engine = TRTEngine("enc_int8_SmoothQ.engine")
    x = torch.randn(96, 3, 224, 224, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out = engine(image=x)  # profile_index=0 by default

    print(engine.input_names, engine.output_names)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
