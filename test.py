import onnx, numpy as np
import onnxruntime as ort
from collections import Counter
# path = "models/0kc5po4ee18_float32_onnx_cuda_msg.onnx"
# #m = onnx.load(path)


# sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
# print("== Inputs ==")
# for i, inp in enumerate(sess.get_inputs()):
#     print(i, "name:", inp.name, "shape:", inp.shape, "dtype:", inp.type)

# print("== Outputs ==")
# for o, out in enumerate(sess.get_outputs()):
#     print(o, "name:", out.name, "shape:", out.shape, "dtype:", out.type)

d = np.load('calib/calib_bev_inputs.npz')
print(d)


# from train.dataloader import RelPosDataModule
# import yaml
# with open('/mnt/beegfs/lchang2/CoViS-Net/train/configs/covisnet.yaml', 'r') as file:
#     config = yaml.safe_load(file)
    
# dataloader = RelPosDataModule(**config["data"])
# dataloader.setup(stage=None)
# t = dataloader.val_dataloader()
# d = next(iter(t[0]))
# import pdb; pdb.set_trace()