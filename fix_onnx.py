import onnx
from onnx import shape_inference

IN_PATH  = "models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp32_cuda_enc.onnx"
OUT_PATH = "model_with_act_etp.onnx"

ACT_NAME = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/act/Mul_1_output_0_QuantizeLinear_Output"

m = onnx.load(IN_PATH)

# (optional) sanity check: does this tensor appear as some node output?
all_node_outputs = {o for n in m.graph.node for o in n.output if o}
if ACT_NAME not in all_node_outputs:
    raise ValueError(f"Tensor name not found as a node output: {ACT_NAME}")

# avoid duplicates if it's already an output
existing_outputs = {o.name for o in m.graph.output}
if ACT_NAME not in existing_outputs:
    # Add as output WITHOUT specifying type/shape (most robust).
    # ONNX allows ValueInfoProto with just a name.
    vi = onnx.ValueInfoProto()
    vi.name = ACT_NAME
    m.graph.output.append(vi)

# Try to populate dtype/shape metadata (nice to have, not strictly required)
try:
    m = shape_inference.infer_shapes(m)
except Exception as e:
    print("Shape inference failed (often OK). Reason:", e)

onnx.save(m, OUT_PATH)
print("Saved:", OUT_PATH)