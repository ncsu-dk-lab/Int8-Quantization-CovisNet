import onnxruntime as ort
import numpy as np

ACT_NAME = "/enc_post/enc_post.0/blocks.0/blocks.0.0/mlp/fc2/input_quantizer/QuantizeLinear_output_0"

sess = ort.InferenceSession("model_with_act.onnx")

# prepare inputs (replace with your real preprocessing)
inputs = {}
for inp in sess.get_inputs():
    # Example: fill only the first input; if your model has multiple inputs, set all of them.
    if inp.name not in inputs:
        shape = [d if isinstance(d, int) else 1 for d in (inp.shape or [1])]
        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

act = sess.run([ACT_NAME], inputs)[0]
print("Activation:", act.shape, act.dtype)
print("First values:", act.reshape(-1)[:10])