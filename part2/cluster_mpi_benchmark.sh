#!/bin/bash

source ./matrices.sh

mkdir -p logs

PARTITION="edu-short"
ACCOUNT="gpu.computing26"
NODELIST="edu01"
TIME_LIMIT="00:05:00"

CPUS_PER_GPU=8
NPROCS_LIST=(1 2 4)

submit_job() {
    local MATRIX="$1"
    local NPROC="$2"

    local NAME
    NAME=$(basename "$MATRIX" .mtx)

    local JOB_NAME="mpi_${NAME}_p${NPROC}"

    echo "Submitting ${JOB_NAME}: ${MATRIX}, ranks=${NPROC}, GPUs=${NPROC}, CPUs=$((NPROC * CPUS_PER_GPU))"

    sbatch --partition="${PARTITION}" \
           --account="${ACCOUNT}" \
           --nodes=1 \
           --ntasks="${NPROC}" \
           --ntasks-per-node="${NPROC}" \
           --gres="gpu:a30.24:${NPROC}" \
           --cpus-per-task="${CPUS_PER_GPU}" \
           --time="${TIME_LIMIT}" \
           --nodelist="${NODELIST}" \
           --job-name="${JOB_NAME}" \
           --output="logs/${JOB_NAME}-%j.out" \
           --error="logs/${JOB_NAME}-%j.err" \
           ./run_cluster.sh "${MATRIX}"
}

for MATRIX in "${real_matrices[@]}"; do
    for NPROC in "${NPROCS_LIST[@]}"; do
        submit_job "${MATRIX}" "${NPROC}"
    done
done

for MATRIX in "${synthetic_matrices[@]}"; do
    for NPROC in "${NPROCS_LIST[@]}"; do
        submit_job "${MATRIX}" "${NPROC}"
    done
done

echo "All jobs submitted."
