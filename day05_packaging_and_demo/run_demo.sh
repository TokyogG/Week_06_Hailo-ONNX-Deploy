#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEF_DEFAULT="${ROOT}/day02_hailo_compile_hef/outputs/resnet18.hef"
IMG_DEFAULT="${ROOT}/day05_packaging_and_demo/assets/test.jpg"

PY="/usr/bin/python3"

usage() {
  cat <<USAGE
Usage:
  ./day05_packaging_and_demo/run_demo.sh [--hef PATH] [--image PATH] [--topk K] [--bench N]

Examples:
  ./day05_packaging_and_demo/run_demo.sh
  ./day05_packaging_and_demo/run_demo.sh --topk 5
  ./day05_packaging_and_demo/run_demo.sh --bench 200

Notes:
- Uses system Python: /usr/bin/python3 (required for pyhailort stability on this setup)
- Passes --no-teardown to avoid a known pyhailort shutdown abort
USAGE
}

HEF="${HEF_DEFAULT}"
IMG="${IMG_DEFAULT}"
TOPK="5"
BENCH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hef) HEF="$2"; shift 2 ;;
    --image) IMG="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --bench) BENCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -x "${PY}" ]]; then
  echo "[ERROR] ${PY} not found/executable. Use system python for HailoRT."
  exit 1
fi

if [[ ! -f "${HEF}" ]]; then
  echo "[ERROR] HEF not found: ${HEF}"
  echo "Expected: ${HEF_DEFAULT}"
  exit 1
fi

if [[ ! -f "${IMG}" ]]; then
  echo "[ERROR] Image not found: ${IMG}"
  echo "Expected: ${IMG_DEFAULT}"
  exit 1
fi

echo "=== Day 05 Demo ==="
echo "Python : ${PY}"
echo "HEF    : ${HEF}"
echo "Image  : ${IMG}"
echo "TopK   : ${TOPK}"
echo "Bench  : ${BENCH}"

if [[ "${BENCH}" -gt 0 ]]; then
  exec "${PY}" "${ROOT}/day05_packaging_and_demo/src/bench_resnet18_hef.py" \
    --hef "${HEF}" \
    --image "${IMG}" \
    --warmup 20 \
    --iters "${BENCH}" \
    --no-teardown
else
  exec "${PY}" "${ROOT}/day05_packaging_and_demo/src/demo_resnet18_hef.py" \
    --hef "${HEF}" \
    --image "${IMG}" \
    --topk "${TOPK}" \
    --debug \
    --no-teardown
fi
