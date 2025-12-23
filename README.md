# Week 06 — Hailo ONNX Deploy (Raspberry Pi 5 + Hailo-8L)

**Hardware:** Raspberry Pi 5 (aarch64) + Hailo-8L (M.2)  
**Runtime:** HailoRT v4.20.0  
**Primary model:** ResNet18  
**Secondary model:** MobileNetV2  

## Goal
Build a repeatable deployment pipeline:
**PyTorch → ONNX → Hailo compile (.hef) → HailoRT inference → benchmarks**

---

## ✅ Day 01 — ONNX Export & Validation (Completed)

**Objective:**  
Establish a *known-good* ONNX model as a stable foundation for Hailo compilation.

**Work completed:**
- Exported pretrained **ResNet18** from PyTorch to ONNX
- Resolved PyTorch 2.x exporter dependency (`onnxscript`)
- Validated ONNX graph integrity using `onnx.checker`
- Verified numerical parity between PyTorch and ONNXRuntime
- Confirmed identical Top-5 predictions

**Key result:**
- `max_abs_diff ≈ 5.2e-06` (FP32 parity within tolerance)
- ONNX model confirmed functionally equivalent to PyTorch

**Artifacts:**
- `day01_env_and_onnx_export/outputs/resnet18.onnx`
- Export and validation scripts committed

---

## Success Metrics
| Item | Target | Result |
|---|---:|---:|
| ONNX export validated | ✅ | **Completed (Day 01)** |
| Hailo compile produces .hef | ✅ | ⏳ |
| End-to-end inference runs on Hailo | ✅ | ⏳ |
| FPS / latency measured | ✅ | ⏳ |
| CPU utilization recorded | ✅ | ⏳ |

---

## Folder Structure
- `day01_env_and_onnx_export/` — export + validate ONNX (ResNet18)
- `day02_hailo_compile_hef/` — compile ONNX → HEF (compiler flags + logs)
- `day03_hailort_inference_runtime/` — minimal HailoRT inference runner
- `day04_benchmarking_and_profiling/` — latency, FPS, system profiling
- `day05_packaging_and_demo/` —_

## Progress Log

### Day 01 — Environment + ONNX Export + Validation
- Created a clean ONNX export pipeline for **ResNet18** (PyTorch → ONNX).
- Resolved missing exporter dependency (`onnxscript`) and validated with:
  - `onnx.checker` load + structural validation
  - PyTorch vs ONNXRuntime parity test
- Result: ONNXRuntime matches PyTorch within tolerance  
  Example parity:
  - `max_abs_diff=5.245209e-06`
  - Top-5 class indices match between PyTorch and ORT.

### Day 02 — Compile ONNX → HEF (Hailo-8L)
- Installed **Hailo Dataflow Compiler (DFC 3.33.0)** on x86 build machine (Python 3.10 venv).
- Parsed ONNX → HAR, optimized with calibration set, compiled to HEF:
  - `resnet18.har` → `resnet18_opt.har` → `resnet18.hef`
- Smoke-tested on Raspberry Pi 5 with HailoRT CLI:
  - **FPS (hw_only): 1149.36**
  - **FPS (streaming): 1150.02**
  - **HW Latency: 2.4446 ms**

---

### Day 03 — HailoRT Runtime Inference (Python)

**Objective:**
Execute **end-to-end runtime inference** on Raspberry Pi 5 using a compiled **Hailo HEF**, validating correct data movement, execution, and output handling via the **HailoRT Python API**.

**Work completed:**

* Implemented a minimal **Python HailoRT inference runner** using `pyhailort`
* Loaded compiled `resnet18.hef` on Raspberry Pi 5 + Hailo-8L
* Queried HEF metadata to confirm runtime I/O expectations:

  * Input: `UINT8`, `NHWC (224×224×3)`
  * Output: `UINT8`, `NC (1000)`
* Built a deterministic preprocessing pipeline:

  * Resize → RGB → `uint8`
  * Contiguous memory
  * Flattened to `(1, 150528)` to match `hw_frame_size`
* Successfully sent input frames to the accelerator
* Retrieved output logits and printed **Top-5 predictions**

**Example runtime output:**

```
=== Top-5 (UINT8 scores) ===
1. class_294   score=249
2. class_296   score=228
3. class_295   score=194
4. class_276   score=192
5. class_270   score=184
```

---

**Key findings / constraints:**

* **System Python required:**
  `pyhailort` is a C++ extension; mixing a Python virtual environment (custom NumPy)
  with system-installed HailoRT caused buffer misreads, bus errors, and segmentation faults.
  Runtime inference must be executed with:

  ```bash
  /usr/bin/python3
  ```

* **Strict input buffer semantics:**
  The runtime is highly sensitive to memory layout:

  * `dtype=uint8`
  * contiguous buffer
  * explicit `(1, 150528)` frame shape
    Shape-correct tensors that violated memory assumptions were rejected by the runtime.

* **Known teardown instability:**
  Inference completes successfully, but some HailoRT Python builds abort during interpreter
  shutdown due to a C++ destructor issue.
  A controlled exit (`os._exit(0)`) is used as a pragmatic workaround to ensure reliable demo execution.

---

**Outcome:**

* Confirmed **fully functional hardware-accelerated inference** using HailoRT Python
* Validated the complete deployment chain:
  **ONNX → HEF → HailoRT → accelerator output**
* Established a reliable baseline runtime script for downstream benchmarking and integration

---

### Day 04 — Benchmarking & Profiling (HailoRT Runtime)

Objective:
Quantify runtime performance of the compiled resnet18.hef on Raspberry Pi 5 + Hailo-8L using both:

HailoRT CLI (hardware baseline)

Python HailoRT runtime (real application path: send → execute → recv)

Work completed:

Created day04_benchmarking_and_profiling/ benchmark workspace

Implemented a repeatable Python benchmark script:

warmup iterations

timed inference loop

latency percentiles (P50 / P95 / P99)

throughput (FPS)

Collected baseline metrics using hailortcli benchmark

Ensured benchmarking was executed via system Python (/usr/bin/python3) to avoid ABI issues

Commands used:

Python benchmark (end-to-end send → recv):

/usr/bin/python3 -X faulthandler day04_benchmarking_and_profiling/src/bench_resnet18_hef.py \
  --hef day02_hailo_compile_hef/outputs/resnet18.hef \
  --image day03_hailort_inference_runtime/assets/test.jpg \
  --warmup 20 \
  --iters 200 \
  --no-teardown


CLI baseline (hardware / driver benchmark):

hailortcli benchmark day02_hailo_compile_hef/outputs/resnet18.hef


Results summary:

HailoRT CLI benchmark

FPS (hw_only): 1149.89

FPS (streaming): 1149.89

HW Latency: 2.445 ms

Python runtime benchmark (send → recv)

Avg latency: 3.049 ms

P50 latency: 2.708 ms

P95 latency: 4.327 ms

P99 latency: 11.769 ms

Throughput: 328.0 FPS

Key findings / interpretation:

CLI numbers represent an upper bound (hardware-focused benchmark).

Python runtime includes additional overhead (buffer handling + synchronization), so:

average latency increases from 2.445 ms (HW) → 3.049 ms (Python E2E)

throughput drops from ~1150 FPS → ~328 FPS

Tail latency (P99) indicates occasional outliers (~11.8 ms), likely due to OS scheduling / runtime jitter.

This becomes a key input for Day 05 packaging and any “real-time” framing.

Outcome:

Established a repeatable benchmarking method for Hailo inference

Captured both:

hardware baseline performance

realistic Python end-to-end performance

Created metrics that can be compared later against:

different models (e.g., MobileNet / YOLO)

different runtimes (C++ vs Python)

different scheduling / batching strategies

