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
