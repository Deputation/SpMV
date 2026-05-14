#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=bench_spmv
#SBATCH --output=bench_spmv-%j.out
#SBATCH --error=bench_spmv-%j.err

module load CUDA/12.5.0

./run_serial.sh
