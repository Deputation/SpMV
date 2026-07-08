#!/bin/bash

set -euo pipefail

MATRIX="$1"
OUTPUT_DIR="${2:-.}"

if [[ ! -x ./baseline_cpu ]]; then
    echo "error: ./baseline_cpu not found or not executable"
    exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "HOSTNAME=$(hostname)"
echo "MATRIX=${MATRIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "START=$(date)"

./baseline_cpu "${MATRIX}" "${OUTPUT_DIR}"

echo "END=$(date)"
