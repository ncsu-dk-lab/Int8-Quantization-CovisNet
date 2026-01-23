import os
import torch
import torch.nn as nn
import tensorrt as trt
from .utils import radius_graph
import torch_geometric
from torch_geometric.utils import dropout_edge
import roma
import torch_scatter
from evaluation.utils import evaluate_model_accruacy
import torch_tensorrt
from typing import Dict
from torch import Tensor
from torch_geometric.utils import dropout_edge, add_self_loops
import modelopt.torch.quantization as mtq

from torch.utils.data import IterableDataset, DataLoader
from .export_model_torchscript_ori import *

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

# ---------- The assembled FP16 model using your engines ----------
class AssembleModel(nn.Module):
    def __init__(
        self,
        comm_range: float = 1000.0,
        gnn_in_channels: int = 24,     # not used inside engines but kept for parity
        gnn_in_seq_len: int = 128,      # expected 128
        bev_gnn_out_channels: int = 384, # expected 384
        dec_out_channels: int = 1,    # expected 1
        engine_dir: str = ".",
        enc_module = None,
        msg_module = None,
        bev_module = None,
        bevdec_module = None,
        post_module = None,
        dtype = torch.float32,
    ):
        super().__init__()
        self.comm_range = comm_range
        self.gnn_in_seq_len = gnn_in_seq_len
        self.bev_gnn_out_channels = bev_gnn_out_channels
        self.enc_module = enc_module
        self.enc_module.eval().to("cuda", dtype=dtype)
        self.msg_module = msg_module
        self.msg_module.eval().to("cuda", dtype=dtype)
        self.bev_module = bev_module
        self.bev_module.eval().to("cuda", dtype=dtype)
        self.bevdec_module = bevdec_module
        self.bevdec_module.eval().to("cuda", dtype=dtype)
        self.post_module = post_module
        self.post_module.eval().to("cuda", dtype=dtype)
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, input):
        bs, n_nodes = input["img_norm"].shape[:2]
        img_flat = input["img_norm"].flatten(0, 1)
        
        graphs = torch_geometric.data.Batch()
        graphs.batch = torch.repeat_interleave(torch.arange(bs), n_nodes, dim=0).to(
            input["img_norm"].device
        )
        graphs.x = self.enc_module(img_flat.to(self.dtype))
        graphs.pos = input["pos"].flatten(0, 1)
        graphs.rot = input["rot_quat"].flatten(0, 1)
        edge_index_pose = radius_graph(#torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )
        graphs.edge_index = edge_index_pose
        x = graphs.x
        edge_index = graphs.edge_index
        edge_preds = self.msg_module(x[edge_index[1]], x[edge_index[0]])

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
    
    # test_loader = DataLoader(DummyBatchStream(device="cuda", bs=8, n_agents=3, num_batches=2),
    #                 batch_size=None,  # important: we're already yielding batches
    #                 num_workers=0)
    #    
    # model = AssembleModel(enc_module=encoder, msg_module=message, bev_module=bev, bevdec_module=bev_dec, post_module=post, dtype=torch.float32)
    # results = evaluate_model_accruacy(model, test_loader)
    # print("!!!!", results)
    
    _, encoder, message, bev, bev_dec, post = load_all_module()




    ## ModelOpt SmoothQuant INT8
    from torch.utils.data import Dataset, DataLoader
    import modelopt.torch.quantization as mtq

    class NpzCalibDataset(Dataset):
        def __init__(self, npz_path, device="cuda"):
            super().__init__()
            self.data = np.load(npz_path)
            self.device = device

            first_array = next(iter(self.data.values()))
            self._len = first_array.shape[0]

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            # Case 1: single input (e.g. imgs)
            if len(self.keys) == 1:
                arr = self.data[self.keys[0]][idx]
                return arr

            # Case 2: multiple inputs (e.g. x_i, x_j)
            sample = {}
            for k in self.keys:
                arr = self.data[k][idx]
                sample[k] = arr
            return sample

            

    def make_calib_loader(npz_path, batch_size=8, device="cuda"):
        dataset = NpzCalibDataset(npz_path, device=device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    device = "cuda"
    encoder = encoder.to(device).eval()

    calib_loader = make_calib_loader("./calib/calib_encoder_inputs.npz",
                                    batch_size=8,
                                    device=device)

    config = mtq.INT8_SMOOTHQUANT_CFG   # INT8 + SmoothQuant

    @torch.no_grad()
    def forward_loop(m):
        m.eval()
        for batch in calib_loader:
            # If dataset returns plain tensors:
            #   batch: [B, 3, H, W] or [B, n_agents, 3, H, W]
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
                # adapt to your forward signature:
                #   e.g. m(batch) or m(image=batch) or m(img_norm=batch)
                _ = m(batch)
            else:
                # If dataset returns dicts (pos / rot_quat / img_norm)
                for k, v in batch.items():
                    batch[k] = v.to(device, non_blocking=True)
                _ = m(**batch)  # or m(batch) depending on your model

    ## Export onnx
    bs = 1
    n_agents = 3
    data = torch.randn(bs, n_agents, 3, resolution, resolution, device=dev)
    img_flat = data.flatten(0, 1).to(torch.float32)

    def export_to_onnx(model, example_inputs, path, input_names, output_names, dynamic_axes):
        """Exports a single PyTorch model to ONNX format."""
        print(f"  Exporting {path}...")
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

    # This will run SmoothQuant + INT8 calibration
    model_int8 = mtq.quantize(encoder, config, forward_loop)
    mtq.print_quant_summary(model_int8)

    export_to_onnx(
        encoder, img_flat, "./models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc.onnx",
        input_names=['image'], output_names=['features'],
        dynamic_axes={'image': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
    )


if __name__ == "__main__":
    main()
