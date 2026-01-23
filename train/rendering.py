import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch_geometric
from matplotlib.patches import ConnectionPatch, Arc, Ellipse
import roma
import os

def polar(angle, dist):
    return torch.stack([angle.sin(), angle.cos()]) * dist


def plot_marker(ax, p, p_var, heading, heading_var, color):
    fov = 120
    dist = 0.75
    fov_half = fov / 2
    var_linestyle = "--"
    fov_half_rad = np.deg2rad(fov_half)
    p_fov_l = p + polar(torch.Tensor([heading + fov_half_rad])[0], dist)
    ax.plot([p[0], p_fov_l[0]], [p[1], p_fov_l[1]], c=color)
    p_fov_r = p + polar(torch.Tensor([heading - fov_half_rad])[0], dist)
    ax.plot([p[0], p_fov_r[0]], [p[1], p_fov_r[1]], c=color)

    if heading_var is not None:
        heading_std = heading_var.sqrt()
        p_var_l = p + polar(torch.Tensor([heading + 2 * heading_std])[0], dist)
        ax.plot(
            [p[0], p_var_l[0]], [p[1], p_var_l[1]], c=color, linestyle=var_linestyle
        )
        p_var_r = p + polar(torch.Tensor([heading - 2 * heading_std])[0], dist)
        ax.plot(
            [p[0], p_var_r[0]], [p[1], p_var_r[1]], c=color, linestyle=var_linestyle
        )
    
    heading = heading.cpu() if isinstance(heading, torch.Tensor) else heading
    arc = Arc(
        p,
        dist * 2,
        dist * 2,
        angle=np.rad2deg(-heading + np.pi / 2),
        theta1=-fov_half,
        theta2=fov_half,
        color=color,
    )
    ax.add_patch(arc)

    if p_var is not None:
        p_std = p_var.sqrt()
        arc_var = Ellipse(
            p,
            2 * p_std[1],
            2 * p_std[0],
            angle=np.rad2deg(0.0),
            color=color,
            fc="none",
            linestyle=var_linestyle,
        )
        ax.add_patch(arc_var)


def _to_np_for_imshow(x, clip01=True):
    """Convert tensor/array to np.float32 or uint8 suitable for plt.imshow."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        # If channel-first (C,H,W) with 1/3/4 channels, move to channels-last
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = x.permute(1, 2, 0)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        x = x.numpy()

    # Cast half → float32 if it came as np.float16
    if x.dtype == np.float16:
        x = x.astype(np.float32)

    # For floats, clamp to [0,1] so grayscale/RGB draw as expected
    if clip01 and np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
    return x


# def render_single(
#     img, bev_label, bev_pred, edge_index, edge_label, edge_pred,
#     save_path=None, dpi=150, show=False
# ):
#     prop_cycle = plt.rcParams["axes.prop_cycle"]
#     colors = prop_cycle.by_key()["color"]
#     n_nodes = img.shape[0]

#     fig, axs = plt.subplot_mosaic(
#         [
#             [f"img_{j}" for j in range(n_nodes)],
#             [f"gt_{j}" for j in range(n_nodes)],
#             [f"pred_{j}" for j in range(n_nodes)],
#         ],
#         figsize=[12, 7],
#     )
#     axs["img_0"].set_ylabel("Image\n\n")
#     axs["gt_0"].set_ylabel("Ground Truth\ny [m]")
#     axs["pred_0"].set_ylabel("Prediction\ny [m]")

#     # Base grids and agent markers on GT/Pred rows
#     for i in range(n_nodes):
#         for ax_label in ["pred", "gt"]:
#             ax = axs[f"{ax_label}_{i}"]
#             ax.set_aspect("equal")
#             ax.set_xlim(-3, 3)
#             ax.set_ylim(-3, 3)
#             ax.grid()
#             plot_marker(ax, torch.zeros(2), None, 0.0, None, colors[i])

#     # Images + BEV overlays
#     for i in range(n_nodes):
#         ax = axs[f"img_{i}"]
#         ax.imshow(_to_np_for_imshow(img[i]))  # robust conversion

#         # Colored frame
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_color(colors[i])
#             spine.set_linewidth(2)

#         axs[f"gt_{i}"].imshow(
#             _to_np_for_imshow(bev_label[i][0]),
#             vmin=0, vmax=1, cmap="gray", alpha=0.5,
#             extent=[-3, 3, -3, 3], interpolation="nearest",
#         )
#         if bev_pred is not None:
#             axs[f"pred_{i}"].imshow(
#                 _to_np_for_imshow(bev_pred[i][0]),
#                 vmin=0, vmax=1, cmap="gray", alpha=0.5,
#                 extent=[-3, 3, -3, 3], interpolation="nearest",
#             )

#         if i > 0:
#             axs[f"pred_{i}"].set_yticklabels([])
#             axs[f"gt_{i}"].set_yticklabels([])
#         axs[f"gt_{i}"].set_xticklabels([])
#         axs[f"pred_{i}"].set_xlabel("x [m]")

#     # Edge labels on GT row
#     if edge_label is not None:
#         for i, j, p, q in zip(
#             edge_index[1],
#             edge_index[0],
#             edge_label["pos"],
#             edge_label["rot"],
#         ):
#             ax = axs[f"gt_{j}"]
#             heading = roma.unitquat_to_rotvec(q)[1]
#             pos = torch.tensor([-p[0], p[2]])
#             plot_marker(ax, pos, None, heading, None, colors[i])

#     # Edge predictions on Pred row (with uncertainty)
#     if edge_pred is not None and not any(v is None for v in edge_pred.values()):
#         for i, j, p_pred, q_pred, p_var, q_var in zip(
#             edge_index[1],
#             edge_index[0],
#             edge_pred["pos"],
#             edge_pred["rot"],
#             edge_pred["pos_var"],
#             edge_pred["rot_var"],
#         ):
#             ax = axs[f"pred_{j}"]
#             heading = roma.unitquat_to_rotvec(q_pred)[1]
#             pos = torch.tensor([-p_pred[0], p_pred[2]])
#             pos_var = torch.tensor([p_var[2], p_var[0]])
#             if pos_var[0] < 1.5 or pos_var[1] < 1.5:
#                 plot_marker(ax, pos, pos_var, heading, q_var[0], colors[i])

#     # Save/show/close
#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

#     return fig

def render_single(
    img, bev_label=None, bev_pred=None, edge_index=None, edge_label=None, edge_pred=None,
    save_path=None, dpi=150, show=False
):
    """
    Render per-node panels:
      - Always an Image row.
      - Optional Ground Truth BEV row (if bev_label is not None).
      - Optional Prediction BEV row (if bev_pred is not None).
      - Optional edge annotations for GT / Pred rows if edge_label / edge_pred provided.
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    n_nodes = img.shape[0]

    rows = ["img"]
    if bev_label is not None:
        rows.append("gt")
    if bev_pred is not None:
        rows.append("pred")

    # Fallback: if neither GT nor Pred is provided, just show images
    if len(rows) == 1:
        height = 3.5
    elif len(rows) == 2:
        height = 5.5
    else:
        height = 7.0

    # Build mosaic spec dynamically
    mosaic = [[f"{row}_{j}" for j in range(n_nodes)] for row in rows]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=[12, height])

    # Left labels only for rows that exist
    if "img" in rows:
        axs["img_0"].set_ylabel("Image\n\n")
    if "gt" in rows:
        axs["gt_0"].set_ylabel("Ground Truth\ny [m]")
    if "pred" in rows:
        axs["pred_0"].set_ylabel("Prediction\ny [m]")

    # Base grids/agent markers on rows that have a BEV plane
    for i in range(n_nodes):
        for ax_label in [r for r in ["gt", "pred"] if r in rows]:
            ax = axs[f"{ax_label}_{i}"]
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid()
            plot_marker(ax, torch.zeros(2), None, 0.0, None, colors[i])

    # Images + optional BEV overlays
    for i in range(n_nodes):
        # Image row
        ax_img = axs[f"img_{i}"]
        ax_img.imshow(_to_np_for_imshow(img[i]))  # robust conversion
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color(colors[i])
            spine.set_linewidth(2)

        # GT overlay
        if "gt" in rows and bev_label is not None:
            # accept either [N,1,H,W] or [N,H,W]
            try:
                gt_slice = bev_label[i][0]
            except Exception:
                gt_slice = bev_label[i]
            axs[f"gt_{i}"].imshow(
                _to_np_for_imshow(gt_slice),
                vmin=0, vmax=1, cmap="gray", alpha=0.5,
                extent=[-3, 3, -3, 3], interpolation="nearest",
            )

        # Pred overlay
        if "pred" in rows and bev_pred is not None:
            try:
                pr_slice = bev_pred[i][0]
            except Exception:
                pr_slice = bev_pred[i]
            axs[f"pred_{i}"].imshow(
                _to_np_for_imshow(pr_slice),
                vmin=0, vmax=1, cmap="gray", alpha=0.5,
                extent=[-3, 3, -3, 3], interpolation="nearest",
            )

        # Axis label/ticks cleanup
        if i > 0:
            if "pred" in rows:
                axs[f"pred_{i}"].set_yticklabels([])
            if "gt" in rows:
                axs[f"gt_{i}"].set_yticklabels([])
        if "gt" in rows:
            axs[f"gt_{i}"].set_xticklabels([])
        if "pred" in rows:
            axs[f"pred_{i}"].set_xlabel("x [m]")
        elif "gt" in rows:
            axs[f"gt_{i}"].set_xlabel("x [m]")

    # Edge labels on GT row
    if bev_label is not None and edge_label is not None and edge_index is not None:
        for i, j, p, q in zip(
            edge_index[1], edge_index[0],
            edge_label.get("pos", []),
            edge_label.get("rot", []),
        ):
            ax = axs[f"gt_{j}"]
            heading = roma.unitquat_to_rotvec(q)[1]
            pos = torch.tensor([-p[0], p[2]])
            plot_marker(ax, pos, None, heading, None, colors[i])

    # Edge predictions on Pred row (with uncertainty)
    have_pred_edges = (
        bev_pred is not None and edge_pred is not None and edge_index is not None and
        all(k in edge_pred for k in ["pos", "rot", "pos_var", "rot_var"])
    )
    if have_pred_edges:
        for i, j, p_pred, q_pred, p_var, q_var in zip(
            edge_index[1], edge_index[0],
            edge_pred["pos"], edge_pred["rot"],
            edge_pred["pos_var"], edge_pred["rot_var"],
        ):
            ax = axs[f"pred_{j}"]
            heading = roma.unitquat_to_rotvec(q_pred)[1]
            pos = torch.tensor([-p_pred[0], p_pred[2]])
            pos_var = torch.tensor([p_var[2], p_var[0]])
            # Skip very uncertain edges
            if pos_var[0] < 1.5 or pos_var[1] < 1.5:
                plot_marker(ax, pos, pos_var, heading, q_var[0], colors[i])

    # Save/show/close
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def unbatch_dict(data, indexes):
    if data is None:
        return None

    unbatched_dict = {}
    for key, value in data.items():
        if value is None:
            unbatched_dict[key] = None
        else:
            unbatched_dict[key] = torch_geometric.utils.unbatch(value, indexes)
    return unbatched_dict


def render_batch(
    datas_batched,
    labels_batched,
    edge_index,
    edge_batch,
    edge_preds,
    node_preds,
    show=False,
):
    n = 16

    labels_pos = labels_batched["pos"].flatten(0, 1)
    pos_sink = labels_pos[edge_index[1]]
    pos_source = labels_pos[edge_index[0]]

    labels_rot = labels_batched["rot_quat"].flatten(0, 1)
    angle_sink = labels_rot[edge_index[1]]
    angle_source = labels_rot[edge_index[0]]

    rel_pos = pos_source - pos_sink
    rel_pos_source = roma.RotationUnitQuat(
        linear=roma.quat_inverse(angle_source)
    ).apply(rel_pos)
    angle_between_nodes = roma.quat_product(angle_source, roma.quat_inverse(angle_sink))
    edge_labels = {
        "pos": rel_pos_source,
        "rot": angle_between_nodes,
    }
    edge_labels_unbatched = unbatch_dict(edge_labels, edge_batch[edge_index[1]])
    edge_preds_unbatched = unbatch_dict(edge_preds, edge_batch[edge_index[1]])
    edge_index_unbatched = torch_geometric.utils.unbatch_edge_index(
        edge_index, edge_batch
    )

    if edge_preds_unbatched is None:
        edge_preds_unbatched = {
            "pos": [None] * len(edge_labels_unbatched),
            "rot": [None] * len(edge_labels_unbatched),
            "pos_var": [None] * len(edge_labels_unbatched),
            "rot_var": [None] * len(edge_labels_unbatched),
        }

    if node_preds is None:
        node_preds = [None] * len(datas_batched["img_raw"])

    renderings = []
    for (
        i,
        img,
        edge_index,
        bev_label,
        edge_label_pos,
        edge_label_rot,
        bev_pred,
        edge_pred_pos,
        edge_pred_rot,
        edge_pred_pos_var,
        edge_pred_rot_var,
    ) in zip(
        range(n),
        datas_batched["img_raw"],
        edge_index_unbatched,
        labels_batched["topdowns_agents"],
        edge_labels_unbatched["pos"],
        edge_labels_unbatched["rot"],
        node_preds,
        edge_preds_unbatched["pos"],
        edge_preds_unbatched["rot"],
        edge_preds_unbatched["pos_var"],
        edge_preds_unbatched["rot_var"],
    ):
        edge_label = {"pos": edge_label_pos, "rot": edge_label_rot}
        edge_pred = {
            "pos": edge_pred_pos,
            "rot": edge_pred_rot,
            "pos_var": edge_pred_pos_var,
            "rot_var": edge_pred_rot_var,
        }
        
        fig = render_single(img, bev_label, bev_pred, edge_index, edge_label, edge_pred)
        
        renderings.append(wandb.Image(fig))
        if show:
            plt.show()
        plt.close()

    return renderings
