import pickle
import statistics
from torchmetrics.classification import BinaryJaccardIndex
from train.dice_loss import DiceLoss
import roma
import torch

def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple([to_device(v, device) for v in obj])
    elif torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, str):
        return obj
    else:
        return obj

def quat_norm_diff(q_a, q_b):
    # https://github.com/utiasSTARS/bingham-rotation-learning/blob/master/quaternions.py
    assert q_a.shape == q_b.shape
    assert q_a.shape[-1] == 4
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a - q_b).norm(dim=1), (q_a + q_b).norm(dim=1)).squeeze()


def quat_chordal_squared_loss(q, q_target, reduce=True):
    # https://github.com/utiasSTARS/bingham-rotation-learning/blob/master/losses.py
    assert q.shape == q_target.shape
    d = quat_norm_diff(q, q_target)
    losses = 2 * d * d * (4.0 - d * d)
    loss = losses.mean() if reduce else losses
    return loss


def quat_gaussian_nll(q, q_target, var, eps=1e-5):
    l = quat_chordal_squared_loss(q, q_target, reduce=False)
    var_eps = torch.maximum(var, torch.tensor(eps))
    return ((var_eps.log() + l / var_eps) / 2.0).mean()


def loss_function(outputs, labels):
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss("binary")

    (edge_preds, node_preds), (edge_index, _) = outputs
    labels_pos = labels["pos"].flatten(0, 1)
    pos_sink = labels_pos[edge_index[1]]
    pos_source = labels_pos[edge_index[0]]

    labels_rot = labels["rot_quat"].flatten(0, 1)
    angle_sink = labels_rot[edge_index[1]]
    angle_source = labels_rot[edge_index[0]]

    rel_pos = pos_source - pos_sink
    rel_pos_source = roma.RotationUnitQuat(
        linear=roma.quat_inverse(angle_source)
    ).apply(rel_pos)
    angle_between_nodes = roma.quat_product(
        angle_source, roma.quat_inverse(angle_sink)
    )

    dist_to_other = torch.norm(rel_pos, p=2, dim=1)

    dist_metric = torch.norm(edge_preds["pos"] - rel_pos_source, dim=1)
    dist_loss = torch.nn.GaussianNLLLoss()(
        edge_preds["pos"], rel_pos_source, edge_preds["pos_var"]
    )

    angle_between_loss = quat_gaussian_nll(
        edge_preds["rot"], angle_between_nodes, edge_preds["rot_var"]
    )

    angle_between_node_gt = roma.unitquat_to_rotvec(angle_between_nodes.float())[
        :, 1
    ]
    angle_between_m = roma.unitquat_geodesic_distance(
        edge_preds["rot"], angle_between_nodes
    )

    bce_loss = 0 # bce_loss_fn(node_preds, labels["topdowns_agents"])
    dice_loss = 0 # dice_loss_fn(node_preds, labels["topdowns_agents"])
    bev_loss = 0 # 0.25 * bce_loss + 0.75 * dice_loss

    return (
        dist_loss,
        angle_between_loss,
        bev_loss,
        {
            "angle_between_m": angle_between_m,
            "angle_between_nodes_gt": angle_between_node_gt,
            "dist_metric": dist_metric,
            "dist": dist_to_other,
            "dist_pred_var": edge_preds["pos_var"],
            "angle_between_nodes_pred_var": edge_preds["rot_var"],
            "bev_bce": bce_loss,
            "bev_dice": dice_loss,
            "rel_pos": rel_pos_source,
            "rel_rot": angle_between_nodes,
        },
    )

@torch.no_grad()
def binary_jaccard(preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    # preds, target: shape (N, ...) with values in [0,1] or {0,1}
    preds_b = (preds >= threshold).to(torch.bool)
    target_b = target.to(torch.bool)
    intersection = (preds_b & target_b).sum().float()
    union = (preds_b | target_b).sum().float()
    return intersection / (union + eps)


def validation_step(module, batch):
    # module = module.to("cuda")
    # module.eval()
    iou = BinaryJaccardIndex()
    with torch.no_grad():
        inputs, labels = batch
        
        test_input_dev_model = {}
        for k, v in inputs.items():
            if k == "img_raw":
                test_input_dev_model[k] = v.to(device="cuda", dtype=torch.float32)
            else:
                test_input_dev_model[k] = to_device(v, "cuda") if torch.is_tensor(v) else v
        
        # outputs = module(
        #     test_input_dev_model["pos"].to("cuda", dtype=torch.float32),
        #     test_input_dev_model["rot_quat"].to("cuda", dtype=torch.float32),
        #     test_input_dev_model["img_norm"].to(device="cuda", dtype=torch.float32),
        # )
        outputs = module(test_input_dev_model)
        outputs = to_device(outputs, "cpu")
        dist_loss, angle_between_loss, bev_loss, meta = loss_function(
            outputs, labels
        )
        loss = dist_loss + angle_between_loss + bev_loss

    fov = 120
    min_overlap = 10
    overlap_threshold = torch.deg2rad(torch.tensor(fov - min_overlap))
    visible_gt = meta["angle_between_nodes_gt"].abs() < overlap_threshold.item()

    ####
    # Filter metrics if required
    dist_metric_visible = meta["dist_metric"][visible_gt]
    angle_between_m_visible = meta["angle_between_m"][visible_gt]

    # Compute median errors
    median_dist_error = torch.median(dist_metric_visible)
    median_angle_error = torch.median(angle_between_m_visible)

    # Convert units
    median_dist_cm = median_dist_error * 100  # Convert meters to centimeters
    median_angle_deg = torch.rad2deg(median_angle_error)  # Convert radians to degrees
    ####

    #iou_value = iou(outputs[0][1].sigmoid(), labels["topdowns_agents"])

    metrics = {
        "val_dice": meta["bev_dice"],
        "val_iou": 0, #iou_value,
        "val_dist_cm": median_dist_cm.item(),
        "val_angle_deg": median_angle_deg.item(),
    }
    return metrics

from statistics import fmean

def _to_float(x):
    import torch
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)

@torch.inference_mode()
def evaluate_model_accruacy(model, dataset):
    """Iterate the whole loader and compute MEAN metrics."""
    print("Evaluating model performance (mean over full val loader)...")

    iou_list, dist_cm_list, angle_deg_list, val_dice_list = [], [], [], []

    # `dataset` is expected to be a DataLoader or any iterable of batches
    for idx, data in enumerate(dataset[0]):
        metrics = validation_step(model, data)  # <-- your existing function
        iou_list.append(_to_float(metrics["val_iou"]))
        dist_cm_list.append(_to_float(metrics["val_dist_cm"]))
        angle_deg_list.append(_to_float(metrics["val_angle_deg"]))
        val_dice_list.append(_to_float(metrics["val_dice"]))

    # Means (handle empty gracefully)
    iou_mean       = fmean(iou_list)       if iou_list else float("nan")
    dist_cm_mean   = fmean(dist_cm_list)   if dist_cm_list else float("nan")
    angle_deg_mean = fmean(angle_deg_list) if angle_deg_list else float("nan")
    val_dice_mean  = fmean(val_dice_list)  if val_dice_list else float("nan")

    print(f"Dice: {val_dice_mean:.4f}, IoU: {iou_mean:.4f}, "
          f"Dist_cm: {dist_cm_mean:.4f}, Angle_deg: {angle_deg_mean:.4f}")

    return {
        "val_dice_mean": val_dice_mean,
        "val_iou_mean": iou_mean,
        "val_dist_cm_mean": dist_cm_mean,
        "val_angle_deg_mean": angle_deg_mean,
    }

def analyze_model_runtime(model, *dummy_data, name: str = ""):
    print(f"{name}: Evaluating model runtime performance...")

    use_cuda = torch.cuda.is_available()
    sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"

    with torch.no_grad():
        # --- Warm up ---
        for _ in range(10):
            model(*dummy_data)
        if use_cuda:
            torch.cuda.synchronize()  # ensure warmup ops finish

        # --- Profile one forward ---
        with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            if use_cuda:
                torch.cuda.synchronize()  # start from a clean slate
            model(*dummy_data)
            if use_cuda:
                torch.cuda.synchronize()  # make sure all kernels complete inside the context

    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))



###### Overwrite radius_graph from geometric library
import torch
import scipy.spatial
import warnings
from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch_geometric

def torch_cluster_radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False
) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        ignore_same_index (bool, optional): If :obj:`True`, each element in
            :obj:`y` ignores the point in :obj:`x` with the same index.
            (default: :obj:`False`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers, ignore_same_index
                                          )


def torch_cluster_radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = torch_cluster_radius(x, x, r, batch, batch,
                        max_num_neighbors,
                        num_workers, batch_size, not loop)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    return torch.stack([row, col], dim=0)



def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    .. warning::

        The CPU implementation of :meth:`radius` with :obj:`max_num_neighbors`
        is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster_radius(x, y, r, batch_x, batch_y,
                                    max_num_neighbors, num_workers)
    return torch_cluster_radius(x, y, r, batch_x, batch_y, max_num_neighbors,
                                num_workers, batch_size)


def radius_graph(
    x: Tensor,
    r: float,
    batch: OptTensor = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Computes graph edges to all points within a given distance.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius_graph

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    .. warning::

        The CPU implementation of :meth:`radius_graph` with
        :obj:`max_num_neighbors` is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if batch is not None and x.device != batch.device:
        warnings.warn(
            "Input tensor 'x' and 'batch' are on different devices "
            "in 'radius_graph'. Performing blocking device transfer",
            stacklevel=2)
        batch = batch.to(x.device)

    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster_radius_graph(x, r, batch, loop, max_num_neighbors,
                                          flow, num_workers)
    return torch_cluster_radius_graph(x, r, batch, loop, max_num_neighbors,
                                      flow, num_workers, batch_size)
