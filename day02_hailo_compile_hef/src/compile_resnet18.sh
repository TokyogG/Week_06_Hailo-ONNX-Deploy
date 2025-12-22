#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

ONNX_IN="../day01_env_and_onnx_export/outputs/resnet18.onnx"
OUT_HEF="outputs/resnet18.hef"

echo "Compiling: ${ONNX_IN}"
echo "Output:    ${OUT_HEF}"

# TODO: replace this with the exact Hailo compile command you use on your setup.
# Keep it explicit + reproducible (flags, calibration steps, etc).
#
# Example placeholders (NOT guaranteed correct for your environment):
# hailo compiler --model "${ONNX_IN}" --output "${OUT_HEF}"

echo "TODO: add hailo compile command here"
