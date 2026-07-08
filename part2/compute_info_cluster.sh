#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=info_spmv
#SBATCH --output=info_spmv-%j.out
#SBATCH --error=info_spmv-%j.err

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

./compute_info.sh
