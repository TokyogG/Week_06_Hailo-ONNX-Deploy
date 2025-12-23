# Week 06 — Hailo ONNX Deploy (Raspberry Pi 5 + Hailo-8L)

**Hardware:** Raspberry Pi 5 (aarch64) + Hailo-8L (M.2)
**Runtime:** HailoRT v4.20.0
**Primary model:** ResNet18
**Secondary model:** MobileNetV2

---

## Goal

Build a repeatable deployment pipeline:

**PyTorch → ONNX → Hailo compile (.hef) → HailoRT inference → benchmarks**

---

## Success Metrics

| Item                               | Target |             Result |
| ---------------------------------- | -----: | -----------------: |
| ONNX export validated              |      ✅ | Completed (Day 01) |
| Hailo compile produces `.hef`      |      ✅ | Completed (Day 02) |
| End-to-end inference runs on Hailo |      ✅ | Completed (Day 03) |
| FPS / latency measured             |      ✅ | Completed (Day 04) |
| Demo packaging complete            |      ✅ | Completed (Day 05) |

---

## Folder Structure

* `day01_env_and_onnx_export/` — Export + validate ONNX (ResNet18)
* `day02_hailo_compile_hef/` — Compile ONNX → HEF (compiler flags + logs)
* `day03_hailort_inference_runtime/` — Minimal HailoRT inference runner
* `day04_benchmarking_and_profiling/` — Latency, FPS, runtime profiling
* `day05_packaging_and_demo/` — One-command demo + benchmark entrypoint

---

## Day 01 — ONNX Export & Validation

**Objective:**
Establish a *known-good* ONNX model as a stable foundation for Hailo compilation.

**Work completed:**

* Exported pretrained **ResNet18** from PyTorch to ONNX
* Resolved PyTorch 2.x exporter dependency (`onnxscript`)
* Validated ONNX graph integrity using `onnx.checker`
* Verified numerical parity between PyTorch and ONNXRuntime
* Confirmed identical Top-5 predictions

**Key result:**

* `max_abs_diff ≈ 5.2e-06` (FP32 parity within tolerance)
* ONNX model confirmed functionally equivalent to PyTorch

**Artifacts:**

* `day01_env_and_onnx_export/outputs/resnet18.onnx`

---

## Day 02 — Compile ONNX → HEF (Hailo-8L)

**Work completed:**

* Installed **Hailo Dataflow Compiler (DFC 3.33.0)** on x86 build machine (Python 3.10 venv)
* Parsed and optimized model:

  * `resnet18.onnx → resnet18.har → resnet18_opt.har → resnet18.hef`
* Deployed HEF to Raspberry Pi 5
* Smoke-tested using HailoRT CLI

**CLI benchmark (baseline):**

* FPS (hw_only): **1149.36**
* FPS (streaming): **1150.02**
* HW Latency: **2.4446 ms**

---

## Day 03 — HailoRT Runtime Inference (Python)

**Objective:**
Execute end-to-end runtime inference on Raspberry Pi 5 using a compiled HEF via the **HailoRT Python API**.

**Work completed:**

* Implemented a minimal Python HailoRT inference runner (`pyhailort`)
* Loaded compiled `resnet18.hef` on Raspberry Pi 5 + Hailo-8L
* Verified HEF runtime I/O expectations:

  * Input: `UINT8`, `NHWC (224×224×3)`
  * Output: `UINT8`, `NC (1000)`
* Built deterministic preprocessing:

  * Resize → RGB → `uint8`
  * Contiguous memory
  * Flattened to `(1, 150528)` to match `hw_frame_size`
* Successfully executed inference and printed Top-5 predictions

**Example output:**

```text
=== Top-5 (UINT8 scores) ===
1. class_294   score=249
2. class_296   score=228
3. class_295   score=194
4. class_276   score=192
5. class_270   score=184
```

### Key constraints discovered

* **System Python required**

  ```bash
  /usr/bin/python3
  ```

  Mixing virtualenv NumPy with system `pyhailort` caused buffer misreads,
  bus errors, and segmentation faults.

* **Strict input buffer semantics**

  * `dtype=uint8`
  * contiguous memory
  * explicit `(1, 150528)` shape

* **Known teardown instability**

  * Inference succeeds, but some builds abort during interpreter shutdown
  * Controlled exit (`os._exit(0)`) used as a pragmatic workaround

**Outcome:**

* Confirmed fully functional hardware-accelerated inference
* Validated complete deployment chain:
  **ONNX → HEF → HailoRT → accelerator output**

---

## Day 04 — Benchmarking & Profiling

**Objective:**
Quantify runtime performance using both:

* **HailoRT CLI** (hardware baseline)
* **Python runtime** (real application path: send → execute → recv)

**Work completed:**

* Implemented repeatable Python benchmark script:

  * warmup iterations
  * timed inference loop
  * latency percentiles (P50 / P95 / P99)
  * throughput (FPS)
* Executed all benchmarks using **system Python**

**Commands used:**

```bash
/usr/bin/python3 day04_benchmarking_and_profiling/src/bench_resnet18_hef.py \
  --hef day02_hailo_compile_hef/outputs/resnet18.hef \
  --image day03_hailort_inference_runtime/assets/test.jpg \
  --warmup 20 \
  --iters 200 \
  --no-teardown
```

```bash
hailortcli benchmark day02_hailo_compile_hef/outputs/resnet18.hef
```

**Results summary:**

*HailoRT CLI*

* FPS (hw_only): **1149.89**
* FPS (streaming): **1149.89**
* HW Latency: **2.445 ms**

*Python runtime (send → recv)*

* Avg latency: **3.049 ms**
* P50 latency: **2.708 ms**
* P95 latency: **4.327 ms**
* P99 latency: **11.769 ms**
* Throughput: **328.0 FPS**

**Interpretation:**

* CLI results represent the hardware upper bound
* Python runtime includes buffer + synchronization overhead
* Tail latency reflects OS scheduling and runtime jitter

---

## Day 05 — Packaging & Demo

**Objective:**
Package Week 06 into a **one-command demo** with optional benchmarking.

**Work completed:**

* Created `day05_packaging_and_demo/` demo workspace
* Added `run_demo.sh` launcher that:

  * enforces system Python
  * validates HEF and image paths
  * supports demo and benchmark modes
* Bundled a sample image for immediate execution
* Encapsulated known teardown workaround (`--no-teardown`)

**Usage:**

```bash
./day05_packaging_and_demo/run_demo.sh
./day05_packaging_and_demo/run_demo.sh --bench 200
```

**Example demo output:**

```text
=== Top-5 (UINT8 scores) ===
1. class_294   score=249.0
2. class_296   score=228.0
3. class_295   score=194.0
4. class_276   score=192.0
5. class_270   score=184.0
```

**Example benchmark (N=200):**

* Avg latency: **2.528 ms**
* P50 latency: **2.526 ms**
* P95 latency: **2.539 ms**
* P99 latency: **2.559 ms**
* Throughput: **395.6 FPS**

**Outcome:**
A portfolio-grade, reproducible demo proving end-to-end deployment:

**HEF → HailoRT runtime → inference output**, with benchmarking built in.

---