import onnx
import numpy as np
from onnx import numpy_helper

# model = onnx.load("models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc.onnx")
model = onnx.load("models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc_scale_patched.onnx")
# Map output tensor name -> producing node
producer = {}
for node in model.graph.node:
    for out in node.output:
        producer[out] = node

def const_value(tensor_name: str):
    node = producer.get(tensor_name, None)
    if node is None:
        return None, "no producer"
    if node.op_type != "Constant":
        return None, f"producer op_type={node.op_type}"
    # Constant has an attribute named "value" (a TensorProto)
    for a in node.attribute:
        if a.name == "value":
            arr = numpy_helper.to_array(a.t)
            return arr, f"Constant dtype={arr.dtype} shape={arr.shape}"
    return None, "Constant without 'value'?"

scale_name = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2/input_quantizer/Constant_1_output_0"
zp_name    = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2/input_quantizer/Constant_output_0"

scale, scale_info = const_value(scale_name)
zp, zp_info       = const_value(zp_name)

print("scale:", scale_info, scale)
print("zero_point:", zp_info, zp)


# import onnx
# import numpy as np
# from onnx import numpy_helper

# def load_onnx(path: str):
#     # If your model uses external data, keep this:
#     return onnx.load(path, load_external_data=True)

# def build_maps(model):
#     # initializer name -> numpy
#     inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

#     # output tensor name -> producer node
#     producer = {}
#     for node in model.graph.node:
#         for out in node.output:
#             producer[out] = node

#     # graph inputs (names)
#     graph_inputs = {i.name for i in model.graph.input}

#     return inits, producer, graph_inputs

# def resolve_tensor(name, inits, producer, graph_inputs):
#     """
#     Return numpy array if resolvable as:
#       - initializer
#       - Constant(value=TensorProto)
#     Else return None + reason.
#     """
#     if name in inits:
#         arr = inits[name]
#         return arr, f"initializer dtype={arr.dtype} shape={arr.shape}"

#     if name in graph_inputs:
#         return None, "graph input (runtime-provided)"

#     node = producer.get(name)
#     if node is None:
#         return None, "no producer node (missing?)"

#     if node.op_type == "Constant":
#         for a in node.attribute:
#             if a.name == "value":
#                 arr = numpy_helper.to_array(a.t)
#                 return arr, f"Constant dtype={arr.dtype} shape={arr.shape}"
#         return None, "Constant but no 'value' attribute"

#     return None, f"produced by op_type={node.op_type} (dynamic)"

# def brief(arr, max_elems=8):
#     if arr is None:
#         return ""
#     flat = arr.flatten()
#     samp = flat[:max_elems]
#     if arr.dtype.kind in "fc":
#         return f"min={float(np.min(flat)):.6g} max={float(np.max(flat)):.6g} sample={samp}"
#     else:
#         return f"min={int(np.min(flat))} max={int(np.max(flat))} sample={samp}"

# def dump_fc2_qdq(model_path: str, fc2_prefix: str):
#     model = load_onnx(model_path)
#     inits, producer, graph_inputs = build_maps(model)

#     # Collect Q/DQ nodes under fc2
#     qdq_nodes = []
#     for n in model.graph.node:
#         if n.op_type in ("QuantizeLinear", "DequantizeLinear") and fc2_prefix in (n.name or ""):
#             qdq_nodes.append(n)

#     # If node names aren't filled, fallback to matching by input/output tensor names
#     if not qdq_nodes:
#         for n in model.graph.node:
#             if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
#                 blob = " ".join(list(n.input) + list(n.output) + ([n.name] if n.name else []))
#                 if fc2_prefix in blob:
#                     qdq_nodes.append(n)

#     print(f"\n=== {model_path} ===")
#     print(f"Matched {len(qdq_nodes)} Q/DQ nodes under: {fc2_prefix}")

#     for n in qdq_nodes:
#         print("\n---")
#         print(f"node: {n.name or '(no name)'}  op={n.op_type}")
#         print(f"  inputs:  {list(n.input)}")
#         print(f"  outputs: {list(n.output)}")

#         # scale is input[1]; zero_point is input[2] if present
#         scale_name = n.input[1] if len(n.input) > 1 else None
#         zp_name    = n.input[2] if len(n.input) > 2 else None

#         if scale_name:
#             scale, info = resolve_tensor(scale_name, inits, producer, graph_inputs)
#             print(f"  scale: {scale_name} -> {info}")
#             if scale is not None:
#                 print(f"         {brief(scale)}")

#         if zp_name:
#             zp, info = resolve_tensor(zp_name, inits, producer, graph_inputs)
#             print(f"  zero_point: {zp_name} -> {info}")
#             if zp is not None:
#                 print(f"             {brief(zp)}")
#         else:
#             print("  zero_point: (missing) -> treated as 0 by ONNX spec")

# if __name__ == "__main__":
#     # Change these paths to your two ONNX files and run twice, or call twice.
#     model_path = "models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp32_cuda_enc.onnx"
#     fc2_prefix = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2"
#     dump_fc2_qdq(model_path, fc2_prefix)