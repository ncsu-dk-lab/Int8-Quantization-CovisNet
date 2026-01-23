from modelopt.onnx.quantization import quantize
import torch
import numpy as np
import pickle
import random
from pathlib import Path

torch.multiprocessing.set_sharing_strategy("file_system")

# Save module-wise npy
from evaluation.export_model_torchscript_ori import (
    LoadedModel, Encoder, Message, PosePost, Bev, BevDecoder
)

print("Initializing submodules ...")
GNN_IN_CH = 24
GNN_IN_SEQ = 128
GNN_OUT_CH = 384
DEVICE = torch.device("cuda")
MODEL_CKPT = Path("models/0kc5po4e/epoch=18-step=452067.ckpt")


encoder = Encoder(GNN_IN_CH, GNN_IN_SEQ).eval().to(DEVICE)
message = Message(GNN_IN_CH, GNN_OUT_CH, GNN_IN_SEQ).eval().to(DEVICE)
post    = PosePost().eval().to(DEVICE)
bev     = Bev(GNN_IN_CH, GNN_OUT_CH, GNN_IN_SEQ).eval().to(DEVICE)
bev_dec = BevDecoder(GNN_OUT_CH, 1).eval().to(DEVICE)

full_model = LoadedModel(GNN_IN_CH, GNN_OUT_CH, GNN_IN_SEQ).eval().to(DEVICE)
full_model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])

# copy trained heads/blocks (mirror your exporter’s pattern)
encoder.enc_post.load_state_dict(full_model.model.enc_post.state_dict())

message.pos_embedding.data.copy_(full_model.model.pose_gnn.pos_embedding.data)
message.aggregator.load_state_dict(full_model.model.pose_gnn.aggregator.state_dict())
message.gnn_decoder_post.load_state_dict(full_model.model.pose_gnn_decoder_post.state_dict())

bev.pos_embedding.data.copy_(full_model.model.bev_gnn.pos_embedding.data)
bev.pose_embedding.load_state_dict(full_model.model.bev_gnn.pose_embedding.state_dict())
bev.aggregator.load_state_dict(full_model.model.bev_gnn.aggregator.state_dict())

bev_dec.decoder.load_state_dict(full_model.model.bev_decoder.state_dict())


import torch, numpy as np, pickle, math
from pathlib import Path
from collections import defaultdict

# ---------- helper ----------
def to_device_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def ensure_img_shape(img):
    # Accept [B, N, 3, 224, 224] or [N, 3, 224, 224]
    if img.ndim == 5:  # [B, N, C, H, W] -> [B*N, C, H, W]
        B, N = img.shape[0], img.shape[1]
        return img.view(B * N, *img.shape[2:]), B, N
    elif img.ndim == 4:
        return img, 1, img.shape[0]  # pretend B=1, N=#agents
    else:
        raise ValueError(f"Unexpected img tensor shape: {img.shape}")

def tensor_list_cat(xs):
    if not xs:
        return None
    return torch.cat(xs, dim=0)

# ---------- paths ----------
out_dir = Path("./calib")
out_dir.mkdir(parents=True, exist_ok=True)

# How many samples to collect per module (change if you want more/less)
MAX_ENC   = 1000    # 1000 # number of images for Encoder
MAX_EDGES = 1000   # 1000 # number of edges (pairs) for Message / PosePost / Bev
MAX_BEV   = 1000    # 1000 # number of bev feature tensors for BevDecoder

# ---------- collectors ----------
enc_imgs   = []
msg_xi     = []
msg_xj     = []
post_in    = []      # edge_preds for PosePost
bev_xi     = []
bev_xj     = []
bev_pose   = []      # same edge_preds for Bev (Pose embedding input)
bevdec_in  = []      # BEV features fed to decoder

# also save extra eval keys (without modification) for later evaluation
eval_meta = defaultdict(list)

# ---------- dataloader ----------
# Expect an existing `dataloader` object (RelPosDataModule) in your env.
# If you already have: train_set = dataloader.train_dataloader(), just reuse it.
from train.dataloader import RelPosDataModule
import yaml
with open('./train/configs/covisnet.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
dataloader = RelPosDataModule(**config["data"])
dataloader.setup(stage=None)
train_loader = dataloader.val_dataloader()

# quick dtype consistency (match your chosen dtype above)
RUN_DTYPE =  torch.float32 # from your model code (fp16 by default)

# ---------- collection loop ----------
enc_count = 0
edge_count = 0
bev_count = 0

full_model.eval()
encoder.eval(); message.eval(); post.eval(); bev.eval(); bev_dec.eval()

with torch.no_grad():
    for batch in iter(train_loader[0]):
        batch = batch[0]
        if enc_count >= MAX_ENC and edge_count >= MAX_EDGES and bev_count >= MAX_BEV:
            break

        batch = to_device_batch(batch, DEVICE)

        # Save the "rest of the keys" verbatim for evaluation later
        # (keep everything — including img_raw and metadata)
        # NOTE: move to cpu before pickling to keep file small-ish
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                eval_meta[k].append(v.detach().cpu())
            else:
                eval_meta[k].append(v)

        # Minimal inputs needed by the model for graph/edges
        # Your full_model expects dict with these keys:
        if not all(x in batch for x in ("pos", "rot_quat", "img_norm")):
            # If your dataloader names differ, adjust here
            continue

        data = {
            "pos":      batch["pos"],
            "rot_quat": batch["rot_quat"],
            "img_norm": batch["img_norm"],
        }

        # 1) Gather Encoder calibration from images
        imgs = data["img_norm"]  # already normalized by your dataset
        imgs_flat, B_eff, N_eff = ensure_img_shape(imgs)

        if enc_count < MAX_ENC:
            need = min(MAX_ENC - enc_count, imgs_flat.shape[0])
            enc_imgs.append(imgs_flat[:need].to(RUN_DTYPE))
            enc_count += need

        # 2) Build edges via the full model to get realistic pairs
        try:
            (edge_outputs, _node_out), (edge_index, _batch_idx) = full_model(data)
        except Exception as e:
            print(f"Skipping batch due to full_model forward error: {e}")
            continue

        # We’ll take a subset of edges per batch for calibration
        # edge_index: shape [2, E], where cols are (src=j, dst=i) in your code
        E = edge_index.shape[1]
        if E == 0:
            continue

        # cap edges in this batch to avoid explosion
        per_batch_cap = min(256, E, MAX_EDGES - edge_count)
        if per_batch_cap <= 0:
            continue

        # Prepare flat image tensor to encode specific agents i/j
        imgs_flat_full = imgs_flat  # [B*N, 3, 224, 224]

        # helper to map (b, agent) index into flat index
        def flat_idx(b, n):
            return b * N_eff + n

        # Heuristic: evenly sample edges across the batch
        step = max(E // per_batch_cap, 1)
        picked = list(range(0, E, step))[:per_batch_cap]

        for e_id in picked:
            if edge_count >= MAX_EDGES and bev_count >= MAX_BEV:
                break

            j = int(edge_index[0, e_id].item())  # src
            i = int(edge_index[1, e_id].item())  # dst

            # Map global indices back to (b, n) pairs
            # In your BEV-GNN, edges are between agents within the same sample.
            # Common packing is: for batch b, agents 0..N_eff-1 map to flat range.
            # We can recover (b, n) via divmod:
            bi, ni = divmod(i, N_eff)
            bj, nj = divmod(j, N_eff)

            # Safety clamps
            if bi >= B_eff or bj >= B_eff:
                continue

            ii = flat_idx(bi, ni)
            jj = flat_idx(bj, nj)

            img_i = imgs_flat_full[ii].unsqueeze(0).to(RUN_DTYPE)
            img_j = imgs_flat_full[jj].unsqueeze(0).to(RUN_DTYPE)

            # Encoder features
            xi = encoder(img_i)  # [1, seq_len, gnn_in_channels]
            xj = encoder(img_j)

            # Message logits (edge preds)
            edge_pred = message(xi, xj)  # [1, 17]

            # PosePost input is edge_pred
            # (PosePost itself runs on CPU/float32 normally; for calibration we just save inputs)
            if edge_count < MAX_EDGES:
                msg_xi.append(xi.detach())
                msg_xj.append(xj.detach())
                post_in.append(edge_pred.detach())
                bev_xi.append(xi.detach())
                bev_xj.append(xj.detach())
                bev_pose.append(edge_pred.detach())
                edge_count += 1

            # BEV feature for decoder
            if bev_count < MAX_BEV:
                bev_feat = bev(xi, xj, edge_pred)  # [1, out_ch]
                bevdec_in.append(bev_feat.detach())
                bev_count += 1

        # (loop next batch)

# ---------- stack & save ----------
def maybe_save_npz(path, **arrays):
    # Drop None and empty
    arrays = {k: v for k, v in arrays.items() if v is not None}
    if not arrays:
        return
    # Convert lists of tensors -> single numpy arrays
    npz_payload = {}
    for k, v in arrays.items():
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            t = tensor_list_cat(v).cpu().numpy()
            npz_payload[k] = t
        elif isinstance(v, torch.Tensor):
            npz_payload[k] = v.cpu().numpy()
        else:
            # already numpy or list of numpy
            npz_payload[k] = v
    np.savez(path, **npz_payload)
    print(f"Saved: {path}")

# Encoder calibration (images)
maybe_save_npz(out_dir / "calib_encoder_inputs.npz", imgs=enc_imgs)

# Message calibration (two inputs)
maybe_save_npz(out_dir / "calib_message_inputs.npz", x_i=msg_xi, x_j=msg_xj)

# PosePost calibration (edge_preds)
maybe_save_npz(out_dir / "calib_posepost_inputs.npz", edge_preds=post_in)

# BEV calibration (x_i, x_j, edge_pred)
maybe_save_npz(out_dir / "calib_bev_inputs.npz", x_i=bev_xi, x_j=bev_xj, edge_preds=bev_pose)

# BevDecoder calibration (bev features)
maybe_save_npz(out_dir / "calib_bevdecoder_inputs.npz", bev_feats=bevdec_in)

# Save eval meta (rest of keys)
meta_path = out_dir / "eval_meta.pkl"
# concat along first dim if tensors; otherwise keep as lists
for k in list(eval_meta.keys()):
    vs = eval_meta[k]
    if all(isinstance(x, torch.Tensor) for x in vs):
        eval_meta[k] = torch.cat(vs, dim=0)
with open(meta_path, "wb") as f:
    pickle.dump(dict(eval_meta), f)
print(f"Saved eval meta to: {meta_path}")

print(
    f"Collected -> enc_imgs: {enc_count}, edges: {edge_count}, bev_feats: {bev_count}"
)
