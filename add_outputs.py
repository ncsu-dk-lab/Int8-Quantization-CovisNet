import argparse
import onnx
from onnx import helper, TensorProto


def find_elem_type(model: onnx.ModelProto, tensor_name: str):
    g = model.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name == tensor_name:
            return vi.type.tensor_type.elem_type
    return None


def add_output(model: onnx.ModelProto, tensor_name: str, elem_type: int):
    g = model.graph
    if any(o.name == tensor_name for o in g.output):
        return

    # IMPORTANT: shape must exist; [] is valid (unknown shape)
    vi = helper.make_tensor_value_info(tensor_name, elem_type, shape=[])
    g.output.append(vi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_onnx", type=str)
    ap.add_argument("output_onnx", type=str)
    ap.add_argument("--tensor", type=str, action="append", required=True)
    ap.add_argument("--dtype", type=str, default=None,
                    help="Force dtype for added outputs: int8|fp16|fp32. "
                         "If omitted, try to infer; fallback fp32.")
    args = ap.parse_args()

    m = onnx.load(args.input_onnx)

    dtype_map = {
        "int8": TensorProto.INT8,
        "fp16": TensorProto.FLOAT16,
        "fp32": TensorProto.FLOAT,
    }
    forced = dtype_map.get(args.dtype.lower()) if args.dtype else None

    for t in args.tensor:
        et = forced
        if et is None:
            et = find_elem_type(m, t)
        if et is None:
            # last resort; ORT can still run if this matches actual runtime dtype,
            # so if you know it's int8, pass --dtype int8.
            et = TensorProto.FLOAT
            print(f"[WARN] Could not infer dtype for '{t}'. Defaulting to FP32. "
                  f"Pass --dtype int8/fp16/fp32 to force.")
        add_output(m, t, et)

    # Shape inference is optional; may fail for Q/DQ graphs, so ignore errors.
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:
        print(f"[WARN] shape_inference failed (still OK): {e}")

    onnx.checker.check_model(m)
    onnx.save(m, args.output_onnx)
    print(f"Saved patched ONNX -> {args.output_onnx}")


if __name__ == "__main__":
    main()