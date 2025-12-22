import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
import torchvision.models as models

ONNX_PATH = os.path.join("outputs", "resnet18.onnx")

def main():
    assert os.path.exists(ONNX_PATH), f"Missing {ONNX_PATH}. Run export first."

    # 1) Structural check (model loads + passes ONNX checker)
    model_onnx = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_onnx)
    print("[OK] ONNX loads + checker passed")

    # 2) Run PyTorch and ONNXRuntime on the SAME input, compare outputs
    torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

    torch.manual_seed(0)
    x_torch = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        y_torch = torch_model(x_torch).cpu().numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    y_ort = sess.run(None, {input_name: x_torch.cpu().numpy()})[0]

    # 3) Numeric parity
    abs_diff = np.max(np.abs(y_torch - y_ort))
    rel_diff = abs_diff / (np.max(np.abs(y_torch)) + 1e-12)

    print(f"[PARITY] max_abs_diff={abs_diff:.6e}  rel={rel_diff:.6e}")

    # Typical tolerance for float32 export
    if abs_diff < 1e-4:
        print("[OK] PyTorch vs ONNXRuntime match within tolerance")
    else:
        print("[WARN] Diff larger than expected; investigate opset / export settings")

    # 4) Smoke test: top-5 indices should match most of the time
    top5_torch = np.argsort(-y_torch[0])[:5]
    top5_ort = np.argsort(-y_ort[0])[:5]
    print("top5_torch:", top5_torch)
    print("top5_ort:  ", top5_ort)

if __name__ == "__main__":
    main()
