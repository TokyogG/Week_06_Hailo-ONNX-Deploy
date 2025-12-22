import os
import torch
import torchvision.models as models

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    out_path = os.path.join(OUT_DIR, "resnet18.onnx")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()