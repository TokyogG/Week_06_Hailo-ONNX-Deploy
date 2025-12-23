#!/usr/bin/env python3
"""
Week 06 — Day 03: HailoRT Runtime Inference (Python)
Goal: load HEF, run inference on an image, print top-5

Known issue on some RPi/HailoRT builds:
  pyhailort may abort at interpreter shutdown ("pure virtual method called").
Workaround:
  --no-teardown  (forces os._exit(0) after printing results)
"""

# NOTE:
# On some HailoRT builds, Python teardown triggers a C++ destructor abort.
# --no-teardown forces a clean exit after inference output.

import argparse
import os
import numpy as np
from PIL import Image

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

INPUT_NAME = "resnet18/input_layer1"
OUTPUT_NAME = "resnet18/fc1"

H, W, C = 224, 224, 3
FRAME_BYTES = H * W * C  # 150528


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_to_input_matrix(image_path: str) -> np.ndarray:
    """
    Returns a uint8 ndarray shaped (1, 150528) to match hw_frame_size=150528.
    This format worked with your pyhailort binding.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)

    nhwc = np.array(img, dtype=np.uint8, copy=True)          # (224,224,3)
    flat = np.ascontiguousarray(nhwc).reshape(1, -1)         # (1,150528)
    flat = np.ascontiguousarray(flat).copy()
    flat.setflags(write=1)

    if flat.shape != (1, FRAME_BYTES):
        raise RuntimeError(f"Bad input tensor shape: {flat.shape} (expected (1,{FRAME_BYTES}))")

    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True, help="Path to .hef")
    ap.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    ap.add_argument("--labels", default=None, help="Optional labels file (one label per line)")
    ap.add_argument("--topk", type=int, default=5, help="Top-K classes to print")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    ap.add_argument(
        "--no-teardown",
        action="store_true",
        help="Workaround pyhailort shutdown abort by exiting immediately after printing results",
    )
    args = ap.parse_args()

    labels = load_labels(args.labels) if args.labels else None
    hef = HEF(args.hef)

    input_tensor = preprocess_to_input_matrix(args.image)

    if args.debug:
        print("Input tensor:", input_tensor.dtype, input_tensor.shape, "nbytes=", input_tensor.nbytes)

    # ---- HailoRT runtime ----
    raw_output = None

    with VDevice() as vdevice:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        configured = vdevice.configure(hef, cfg)[0]

        in_params = InputVStreamParams.make_from_network_group(
            configured, quantized=True, format_type=FormatType.UINT8
        )
        out_params = OutputVStreamParams.make_from_network_group(
            configured, quantized=True, format_type=FormatType.UINT8
        )

        # Your build requires activation as a context manager
        with configured.activate():
            in_vs = configured._create_input_vstreams(in_params)
            out_vs = configured._create_output_vstreams(out_params)

            try:
                input_stream = in_vs.get_input_by_name(INPUT_NAME)
                output_stream = out_vs.get_output_by_name(OUTPUT_NAME)

                if args.debug:
                    print("Input stream methods:", [m for m in ("send", "write") if hasattr(input_stream, m)])
                    print("Output stream methods:", [m for m in ("recv", "read") if hasattr(output_stream, m)])

                input_stream.send(input_tensor)
                raw_output = output_stream.recv()

            finally:
                # Explicit closes to reduce likelihood of teardown crash
                try:
                    out_vs.close()
                except Exception:
                    pass
                try:
                    in_vs.close()
                except Exception:
                    pass

    # ---- Post-process ----
    scores_u8 = np.array(raw_output, dtype=np.uint8).reshape(-1)
    scores = scores_u8.astype(np.float32, copy=False)

    k = min(args.topk, scores.size)
    top_idx = np.argsort(scores)[-k:][::-1]

    print(f"\n=== Top-{k} (UINT8 scores) ===")
    for rank, i in enumerate(top_idx, 1):
        name = labels[i] if (labels and i < len(labels)) else f"class_{i}"
        print(f"{rank:>2}. {name:<40} score={scores[i]:.1f}")

    # Workaround: some pyhailort builds abort at interpreter shutdown.
    if args.no_teardown:
        os._exit(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
