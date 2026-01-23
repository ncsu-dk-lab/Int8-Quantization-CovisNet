import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from train.rendering import render_single


# def load_img(path):
#     transform = transforms.Compose(
#         [
#             transforms.PILToTensor(),
#             # RandomCenterCrop(0.5),
#             transforms.Resize(
#                 224,
#                 antialias=True,
#                 interpolation=transforms.InterpolationMode.BILINEAR,
#             ),
#             transforms.CenterCrop(224),
#             transforms.ConvertImageDtype(torch.half),
#         ]
#     )
#     img = Image.open(path)
#     return transform(img)


# scene_paths = [
#     [
#         "datasets/dataset_real_5_231024/intellab_01/sensor_0/image_proc/02351.jpg",
#         "datasets/dataset_real_5_231024/intellab_01/sensor_2/image_proc/02479.jpg",
#         "datasets/dataset_real_5_231024/intellab_01/sensor_2/image_proc/02491.jpg",
#     ],
#     [
#         "datasets/dataset_real_5_231024/sn-corridor_01/sensor_0/image_proc/00977.jpg",
#         "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/01412.jpg",
#     ],
#     [
#         "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03431.jpg",
#         "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03377.jpg",
#         "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03407.jpg",
#     ],
#     [
#         "datasets/dataset_real_5_231024/sn05_01/sensor_1/image_proc/03052.jpg",
#         "datasets/dataset_real_5_231024/sn05_01/sensor_1/image_proc/00562.jpg",
#         "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/01038.jpg",
#     ],
#     [
#         "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/04189.jpg",
#         "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/00484.jpg",
#         "datasets/dataset_real_5_231024/sn05_01/sensor_0/image_proc/01057.jpg",
#     ],
# ]

from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_img(path):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),   # 0..1
        transforms.Resize(224, antialias=True,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

# --- Build scene_paths from three folders (sensor0, sensor1, sensor2) ---
SENSOR_DIRS = [Path("frames/sensor0"), Path("frames/sensor1"), Path("frames/sensor2")]
FRAME_START, FRAME_END = 1, 10  # frame_000001.png ... frame_000010.png

scene_paths = []
for i in range(FRAME_START, FRAME_END + 1):
    fname = f"frame_{i:06d}.png"
    triplet = [str(d / fname) for d in SENSOR_DIRS]
    scene_paths.append(triplet)



def run(model_base):
    enc = torch.jit.load(f"models_exported/{model_base}_float16_jit_cuda_enc.ts")
    msg = torch.jit.load(f"models_exported/{model_base}_float16_jit_cuda_msg.ts")
    post = torch.jit.load(f"models_exported/{model_base}_float32_jit_cpu_post.ts")
    bev = torch.jit.load(f"models_exported/{model_base}_float16_jit_cuda_bev.ts")
    bev_dec = torch.jit.load(f"models_exported/{model_base}_float16_jit_cuda_bevdec.ts")

    for scene_path in scene_paths:
        data = {
            "img": torch.stack(
                [load_img(path) for i, path in enumerate(scene_path)], dim=0
            ),
            "encs": None,
            "edge_index": [[], []],
            "edge_preds": {
                "msg": [],
                "pos": [],
                "rot": [],
                "pos_var": [],
                "rot_var": [],
            },
            "node_preds": [],
        }
        with torch.no_grad():
            data["encs"] = [enc(img.unsqueeze(0).to("cuda", dtype=torch.half)) for img in data["img"]]
            for i, enc_i in enumerate(data["encs"]):
                for j, enc_j in enumerate(data["encs"]):
                    if i == j:
                        continue
                    data["edge_index"][0].append(j)
                    data["edge_index"][1].append(i)

                    m = msg(enc_i.to("cuda", dtype=torch.half), enc_j.to("cuda", dtype=torch.half))
                    data["edge_preds"]["msg"].append(m)

                    pos, pos_var, heading, heading_var = post(m.to("cuda", dtype=torch.float))
                    data["edge_preds"]["pos"].append(pos[0])
                    data["edge_preds"]["rot"].append(heading[0])
                    data["edge_preds"]["pos_var"].append(pos_var[0])
                    data["edge_preds"]["rot_var"].append(heading_var)

                agg = bev(enc_i.to("cuda", dtype=torch.half), enc_i.to("cuda", dtype=torch.half), torch.zeros(1, 17, device="cuda", dtype=torch.half))
                edge_index = torch.tensor(data["edge_index"])
                edge_msg = torch.cat(data["edge_preds"]["msg"], dim=0)
                for j, edge_msg in zip(
                    edge_index[0][edge_index[1] == i], edge_msg[edge_index[1] == i]
                ):
                    agg += bev(enc_i.to("cuda", dtype=torch.half), data["encs"][j].to("cuda", dtype=torch.half), edge_msg.unsqueeze(0).to("cuda", dtype=torch.half))
                data["node_preds"].append(bev_dec(agg.to("cuda", dtype=torch.half)))
        bev_pred = torch.cat(data["node_preds"], dim=0)
        bev_label = torch.zeros_like(bev_pred)
        # fig = render_single(
        #     data["img"].cpu(),
        #     bev_label.cpu(),
        #     bev_pred.cpu(),
        #     data["edge_index"],
        #     None,
        #     data["edge_preds"],
        #     save_path=f"images/{str(scene_path)}.png", dpi=150, show=False
        # )
        fig = render_single(
            img=data["img"].cpu(),                # [N, C, H, W]
            bev_label=None,                   # no GT
            bev_pred=bev_pred.cpu(),                # [N, 1, H_bev, W_bev] or [N, H_bev, W_bev]
            edge_index=data["edge_index"],
            edge_label=None,                  # no GT edges
            edge_pred=data["edge_preds"],
            save_path=f"images/{str(scene_path)}.png",
            dpi=150, show=False
        )

        fig.savefig("images/example1.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run("0kc5po4ee18")
