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
from typing import Dict
from torch import Tensor

attn = Attention
dtype = torch.float32


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

        # dtype = next(self.enc.parameters()).dtype
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


class BevDecoder(torch.nn.Module):
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


out_base_dir = Path("./models")
model_in_file = Path("models/0kc5po4e/epoch=18-step=452067.ckpt")
device_str = "cuda"
dev = torch.device(device_str)
gnn_in_channels = 24
gnn_in_seq_len = 128
gnn_out_channels = 384
resolution = 224

model = LoadedModel(
    gnn_in_channels, gnn_out_channels, gnn_in_seq_len
).eval().to(dev, dtype=dtype)
model.load_state_dict(torch.load(model_in_file, weights_only=False)["state_dict"])

model_inp = {
    "img_norm": torch.rand(10, 5, 3, resolution, resolution, device=dev, dtype=dtype),
    "pos": torch.rand(10, 5, 3),
    "rot_quat": torch.rand(10, 5, 4),
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

bev_dec = BevDecoder(gnn_out_channels, 1).eval().to(dev)
bev_dec.decoder.load_state_dict(model.model.bev_decoder.state_dict())

bs = 1
n_agents = 3
data = {
    "pos": torch.rand(bs, n_agents, 3, device=dev, dtype=dtype) * 8,
    "rot_quat": torch.rand(bs, n_agents, 4, device=dev, dtype=dtype),
    "img_norm": torch.randn(bs, n_agents, 3, resolution, resolution, device=dev, dtype=dtype) * 3,
}

(model_out_edges, model_out_nodes), (edge_index, batch) = model(data)

def model_decentr_forward(data, edge_index, f_enc, f_msg, f_post, dtype):
    f_enc.eval()
    f_msg.eval()
    f_post.eval()
    out = []
    img_flat = data["img_norm"].flatten(0, 1).to(dtype)
    for i, j in zip(edge_index[1], edge_index[0]):
        img_i = img_flat[i].unsqueeze(0)
        img_j = img_flat[j].unsqueeze(0)
        x_i = f_enc.to(dtype)(img_i)
        x_j = f_enc.to(dtype)(img_j)
        aggr = f_msg.to(dtype)(x_i, x_j)
        p = f_post.to(torch.float)(aggr.to(torch.float))
        out_cat = torch.concatenate(
            [c.unsqueeze(1) if c.ndim == 1 else c for c in p], dim=1
        )
        out.append(out_cat)
    return torch.concatenate(out, dim=0)


def eval_outputs(a, b, dtype=torch.float):
    def merge_dict(v):
        return torch.concatenate(
            [v[k].unsqueeze(1) if v[k].ndim == 1 else v[k] for k in v.keys()], dim=1
        )

    err = a.to(dtype) - merge_dict(b).to(dtype)
    serr = err**2
    abserr = err.abs()
    mse = serr.mean()
    mae = abserr.mean()
    print("mse", mse.item(), "mae", mae.item())
    return mse.item()


def model_trace(data, f_enc, f_msg, f_post, f_bev, f_bev_dec, dtype=torch.float):
    out = []
    img_flat = data["img_norm"].to(dtype).flatten(0, 1)
    i, j = 0, 0
    img_i = img_flat[i].unsqueeze(0)
    img_j = img_flat[j].unsqueeze(0)
    x_i = f_enc.to(dtype)(img_i)
    x_j = f_enc.to(dtype)(img_j)
    aggr = f_msg.to(dtype)(x_i, x_j)
    bev = f_bev.to(dtype)(x_i, x_j, aggr)
    bev_dec = f_bev_dec.to(dtype)(bev)

    def trace_with_label(label, mod, inputs, *, dtype=None, check_trace=True):
        if dtype is not None:
            mod = mod.to(dtype)
            if isinstance(inputs, tuple):
                inputs = tuple(t.to(dtype) for t in inputs)
            else:
                inputs = inputs.to(dtype)
        print(f"[TRACE] starting {label}")
        out = torch.jit.trace(mod, inputs, check_trace=check_trace)
        print(f"[TRACE] ok: {label}")
        return out

    f_enc_traced  = trace_with_label("enc",  f_enc,  img_i,              dtype=dtype)
    f_msg_traced  = trace_with_label("msg",  f_msg,  (x_i, x_j),          dtype=dtype)
    f_post_traced = trace_with_label("post", f_post, aggr.to(torch.float), dtype=None)
    f_bev_traced  = trace_with_label("bev",  f_bev,  (x_i, x_j, aggr),    dtype=dtype)
    f_bev_dec_traced = trace_with_label("bev_dec", f_bev_dec, bev,        dtype=dtype)

    return f_enc_traced, f_msg_traced, f_post_traced, f_bev_traced, f_bev_dec_traced


model_decentr_out = model_decentr_forward(
    data, edge_index, encoder, message, post, dtype=dtype
)
mse = eval_outputs(model_decentr_out, model_out_edges)


def convert(dtype=torch.float):
    f_enc_traced, f_msg_traced, f_post_traced, f_bev_traced, f_bev_dec_traced = (
        model_trace(data, encoder, message, post, bev, bev_dec, dtype=dtype)
    )

    model_jit_decentr_out = model_decentr_forward(
        data, edge_index, f_enc_traced, f_msg_traced, f_post_traced, dtype=dtype
    )
    mse = eval_outputs(model_jit_decentr_out, model_out_edges, dtype=dtype)
    # assert mse < 1e-10

    model_version = model_in_file.parent.stem.replace(":", "-")
    model_epoch = int(model_in_file.stem.split("-")[0].split("=")[1])
    model_prefix = f"{model_version}e{model_epoch:02d}"

    type_prefix = str(dtype).split(".")[1]

    torch.jit.save(
        f_enc_traced,
        out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_enc.ts",
    )
    torch.jit.save(
        f_msg_traced,
        out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_msg.ts",
    )
    torch.jit.save(
        f_post_traced, out_base_dir / f"{model_prefix}_float32_jit_{device_str}_post.ts"
    )
    torch.jit.save(
        f_bev_traced,
        out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_bev.ts",
    )
    torch.jit.save(
        f_bev_dec_traced,
        out_base_dir / f"{model_prefix}_{type_prefix}_jit_{device_str}_bevdec.ts",
    )


convert(dtype)
# convert(torch.half)