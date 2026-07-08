#!/bin/bash

mkdir -p logs

MATRIX="dataset/degme/degme.mtx"

PARTITION="edu-short"
ACCOUNT="gpu.computing26"
NODELIST="edu01"
TIME_LIMIT="00:05:00"

CPUS_PER_GPU=8
NPROC=2

NAME=$(basename "$MATRIX" .mtx)
JOB_NAME="mpi_${NAME}_p${NPROC}_test"

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
