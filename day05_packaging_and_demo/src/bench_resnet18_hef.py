#!/usr/bin/env python3
"""
Day 04 — Benchmarking & Profiling (HailoRT Python)

- Uses system Python (/usr/bin/python3) to avoid ABI clashes.
- Reuses the Day03 working pipeline: UINT8 input, (1,150528) frame.
- Adds warmup + timed loop + latency stats.
- Optional --no-teardown to bypass pyhailort shutdown abort.
"""

import argparse
import os
import time
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


def preprocess_to_input_matrix(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)

    nhwc = np.array(img, dtype=np.uint8, copy=True)          # (224,224,3)
    flat = np.ascontiguousarray(nhwc).reshape(1, -1)         # (1,150528)
    flat = np.ascontiguousarray(flat).copy()
    flat.setflags(write=1)

    if flat.shape != (1, FRAME_BYTES):
        raise RuntimeError(f"Bad input tensor shape: {flat.shape} (expected (1,{FRAME_BYTES}))")

    return flat


def percentile(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not timed)")
    ap.add_argument("--iters", type=int, default=200, help="Timed iterations")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-teardown", action="store_true")
    args = ap.parse_args()

    hef = HEF(args.hef)
    input_tensor = preprocess_to_input_matrix(args.image)

    if args.debug:
        print("Input tensor:", input_tensor.dtype, input_tensor.shape, "nbytes=", input_tensor.nbytes)

    lat_ms = []

    with VDevice() as vdevice:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        configured = vdevice.configure(hef, cfg)[0]

        in_params = InputVStreamParams.make_from_network_group(
            configured, quantized=True, format_type=FormatType.UINT8
        )
        out_params = OutputVStreamParams.make_from_network_group(
            configured, quantized=True, format_type=FormatType.UINT8
        )

        with configured.activate():
            in_vs = configured._create_input_vstreams(in_params)
            out_vs = configured._create_output_vstreams(out_params)

            try:
                input_stream = in_vs.get_input_by_name(INPUT_NAME)
                output_stream = out_vs.get_output_by_name(OUTPUT_NAME)

                # Warmup (stabilize runtime)
                for _ in range(args.warmup):
                    input_stream.send(input_tensor)
                    _ = output_stream.recv()

                # Timed loop (per-iteration latency)
                for _ in range(args.iters):
                    t0 = time.perf_counter_ns()
                    input_stream.send(input_tensor)
                    _ = output_stream.recv()
                    t1 = time.perf_counter_ns()
                    lat_ms.append((t1 - t0) / 1e6)

            finally:
                # Explicit close to reduce teardown crash odds
                try:
                    out_vs.close()
                except Exception:
                    pass
                try:
                    in_vs.close()
                except Exception:
                    pass

    lat = np.array(lat_ms, dtype=np.float64)
    avg = float(lat.mean())
    p50 = percentile(lat, 50)
    p95 = percentile(lat, 95)
    p99 = percentile(lat, 99)
    fps = 1000.0 / avg if avg > 0 else 0.0

    print("\n=== HailoRT Python Benchmark (send→recv) ===")
    print(f"Warmup iters: {args.warmup}")
    print(f"Timed iters : {args.iters}")
    print(f"Avg latency : {avg:.3f} ms")
    print(f"P50 latency : {p50:.3f} ms")
    print(f"P95 latency : {p95:.3f} ms")
    print(f"P99 latency : {p99:.3f} ms")
    print(f"Throughput  : {fps:.1f} FPS")

    if args.csv:
        # CSV: iter,latency_ms
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("iter,latency_ms\n")
            for i, v in enumerate(lat_ms):
                f.write(f"{i},{v:.6f}\n")
        print(f"\nSaved CSV: {args.csv}")

    if args.no_teardown:
        os._exit(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
