import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load("models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc.onnx")

scale_name = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2/input_quantizer/Constant_1_output_0"
new_scale_val = 0.030454890802502632

# Build producer map (same as yours)
producer = {}
for node in model.graph.node:
    for out in node.output:
        producer[out] = node

scale_node = producer.get(scale_name, None)
if scale_node is None:
    raise RuntimeError(f"No producer node for {scale_name}")
if scale_node.op_type != "Constant":
    raise RuntimeError(f"Producer is {scale_node.op_type}, not Constant")

# Find Constant's TensorProto attribute "value" and overwrite it
for a in scale_node.attribute:
    if a.name == "value":
        old = numpy_helper.to_array(a.t)
        # preserve dtype/shape (often scalar float)
        new_arr = np.array(new_scale_val, dtype=old.dtype).reshape(old.shape)
        a.t.CopyFrom(numpy_helper.from_array(new_arr))
        print("Old scale:", old, " New scale:", new_arr)
        break
else:
    raise RuntimeError("Constant node has no 'value' attribute")

onnx.checker.check_model(model)
onnx.save(model, "models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc_scale_patched.onnx")
print("Saved patched model.")