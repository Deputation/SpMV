#!/bin/bash

MATRIX="$1"

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "Matrix: ${MATRIX}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Tasks/ranks: ${SLURM_NTASKS}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

echo "Binary:"
ls -lh ./output/mpi_spmv

echo "MPI launcher:"
which mpirun
mpirun --version

echo "GPUs visible to job:"
nvidia-smi -L

echo "About to launch mpi_spmv"
date

mpirun -np "${SLURM_NTASKS}" ./output/mpi_spmv "${MATRIX}"

echo "Finished mpi_spmv"
date
