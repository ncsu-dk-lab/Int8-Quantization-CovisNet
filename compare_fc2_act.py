import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model_with_act_etp.onnx")
#ACT_NAME = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2/input_quantizer/QuantizeLinear_output_0"
ACT_NAME = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/act/Mul_1_output_0_QuantizeLinear_Output"


# prepare inputs (replace with your real preprocessing)
inputs = {}
for inp in sess.get_inputs():
    # Example: fill only the first input; if your model has multiple inputs, set all of them.
    if inp.name not in inputs:
        shape = [d if isinstance(d, int) else 1 for d in (inp.shape or [1])]
        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

act = sess.run([ACT_NAME], inputs)[0]
a = act.astype(np.int32).reshape(-1)
print("Activation:", act.shape, act.dtype)
print("First values:", act.reshape(-1)[:10])

# Basic stats on int8 values
print("int8 min/max:", a.min(), a.max())
print("int8 mean/std:", a.mean(), a.std())

# Saturation / clipping rate (both ends)
sat_pos = np.mean(a == 127)
sat_neg = np.mean(a == -128)
sat_any = np.mean((a == 127) | (a == -128))
print(f"saturation: +127 {sat_pos*100:.4f}% | -128 {sat_neg*100:.4f}% | total {sat_any*100:.4f}%")

# Value distribution (how “spread” it is)
abs_a = np.abs(a)
for p in [50, 90, 95, 99, 99.9]:
    print(f"|q| p{p}: {np.percentile(abs_a, p):.2f}")



scale = 0.030454890802502632 #0.007874016
zp = 0               # change if your model uses non-zero zp

# dequantize to float for real-value stats
x = (a - zp) * scale

print("real min/max:", x.min(), x.max())
print("real mean/std:", x.mean(), x.std())

abs_x = np.abs(x)
for p in [50, 90, 95, 99, 99.9]:
    print(f"|x| p{p}: {np.percentile(abs_x, p):.6f}")

# How close you are to the representable real range
max_real = 127 * scale
min_real = -128 * scale
print("representable real range:", min_real, max_real)