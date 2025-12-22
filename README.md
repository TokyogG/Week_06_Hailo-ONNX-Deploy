# Week_06_ONNX-Hailo-Deploy

This folder contains all deliverables for **Week_06_ONNX-Hailo-Deploy**  
More details will be added as the bootcamp progresses.

---
Generated automatically on: Mon Dec  1 01:43:18 PM EST 2025
# Week 06 — Hailo ONNX Deploy (Raspberry Pi 5 + Hailo-8L)

**Hardware:** Raspberry Pi 5 (aarch64) + Hailo-8L (M.2)  
**Runtime:** HailoRT v4.20.0  
**Primary model:** ResNet18  
**Secondary model:** MobileNetV2  

## Goal
Build a repeatable deployment pipeline:
**PyTorch → ONNX → Hailo compile (.hef) → HailoRT inference → benchmarks**

## Success Metrics (fill as you go)
| Item | Target | Result |
|---|---:|---:|
| ONNX export validated | ✅ |  |
| Hailo compile produces .hef | ✅ |  |
| End-to-end inference runs on Hailo | ✅ |  |
| FPS / latency measured | ✅ |  |
| CPU utilization recorded | ✅ |  |

---

## Folder Structure
- `day01_env_and_onnx_export/` — export + validate ONNX (resnet18 + mobilenet_v2)
- `day02_hailo_compile_hef/` — compile ONNX → HEF (and record compiler flags)
- `day03_hailort_inference_runtime/` — minimal HailoRT runner (image → inference → topk)
- `day04_benchmarking_and_profiling/` — FPS/latency, perf counters, power notes
- `day05_packaging_and_demo/` — “one command demo”, clean outputs, final writeup
- `models/` — model artifacts (keep large files out of git if needed)
- `scripts/` — helper scripts (setup checks, download, run)
- `notes/` — troubleshooting + learnings

---

## Quickstart
### 0) Sanity check environment
```bash
python3 --version
hailortcli --version
