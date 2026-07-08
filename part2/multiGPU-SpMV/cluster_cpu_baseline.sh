#!/bin/bash

set -euo pipefail

source ./baseline_matrices.sh

mkdir -p logs

PARTITION="edu-short"
ACCOUNT="gpu.computing26"
NODELIST="edu01"
TIME_LIMIT="00:05:00"

CPUS_PER_TASK=8
GPUS_REQUESTED=1
OUTPUT_DIR="."

submit_job() {
    local MATRIX="$1"

    local NAME
    NAME=$(basename "$MATRIX" .mtx)

    local JOB_NAME="cpu_base_${NAME}"

    echo "Submitting ${JOB_NAME}: ${MATRIX}, CPUs=${CPUS_PER_TASK}, GPUs=${GPUS_REQUESTED}"

    sbatch --partition="${PARTITION}" \
           --account="${ACCOUNT}" \
           --nodes=1 \
           --ntasks=1 \
           --ntasks-per-node=1 \
           --gres="gpu:a30.24:${GPUS_REQUESTED}" \
           --cpus-per-task="${CPUS_PER_TASK}" \
           --time="${TIME_LIMIT}" \
           --nodelist="${NODELIST}" \
           --job-name="${JOB_NAME}" \
           --output="logs/${JOB_NAME}-%j.out" \
           --error="logs/${JOB_NAME}-%j.err" \
           ./run_cpu_baseline_cluster.sh "${MATRIX}" "${OUTPUT_DIR}"
}

for MATRIX in "${real_matrices[@]}"; do
    submit_job "${MATRIX}"
done

for MATRIX in "${synthetic_matrices[@]}"; do
    submit_job "${MATRIX}"
done

echo "All CPU baseline jobs submitted."
