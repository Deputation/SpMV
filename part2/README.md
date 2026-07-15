# Multi-GPU SpMV

This repository has been developed by the student Giuseppe Screnci.

This repository contains the code and scripts used for the Deliverable 2 multi-GPU sparse matrix-vector multiplication experiments.

The implementation benchmarks distributed SpMV on one multi-GPU node using MPI, CUDA-aware MPI, local COO/CSR matrices, ghost-vector exchange, custom CUDA kernels, CPU kernels, and cuSPARSE kernels.

The cluster scripts are configured for the course cluster environment.

## Requirements

On the cluster, the build script loads:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
```

The benchmark scripts submit jobs to the configured SLURM partition/account/node.

## Multi-GPU benchmark workflow

From the root of this repository, run:

```bash
./download_part2_dataset.sh
./compile_cluster.sh
./generate_synthetic_dataset.sh
./cluster_mpi_benchmark.sh
```

Wait for the submitted SLURM jobs to finish.

Then merge the generated benchmark CSVs:

```bash
./merge.sh
```

The merged MPI benchmark results are written to:

```text
merged_results/
├── 1_compute.csv
├── 1_parsing.csv
├── 2_compute.csv
├── 2_parsing.csv
├── 4_compute.csv
└── 4_parsing.csv
```

The raw per-job outputs are written under:

```text
results/
logs/
```

## What the multi-GPU scripts do

`download_part2_dataset.sh` downloads and extracts the real matrices used for Deliverable 2 into `dataset/`.

`compile_cluster.sh` cleans the build, loads CUDA and CUDA-aware OpenMPI modules, and builds the project.

`generate_synthetic_dataset.sh` creates the synthetic 10M, 20M, and 40M nonzero matrices, including balanced and unbalanced versions, under `synthetic_dataset/`.

`cluster_mpi_benchmark.sh` submits one SLURM job per matrix and per rank count. It runs with 1, 2, and 4 MPI ranks, requesting the same number of A30 GPUs.

`merge.sh` concatenates the per-job MPI CSVs into rank-specific merged CSVs in `merged_results/`.

## CPU baseline workflow

The provided CPU baseline is built and run from a different directory:

```bash
cd multiGPU-SpMV
./compile.sh
./cluster_cpu_baseline.sh
```

Wait for the submitted SLURM jobs to finish.

Then merge the scattered CPU baseline CSVs:

```bash
./merge_cpu_baseline.sh
```

This produces:

```text
baseline_cpu_merged.csv
```

## Optional matrix information

To compute matrix structural information on the cluster:

```bash
sbatch compute_info_cluster.sh
```

This runs the matrix information executable on the real and synthetic matrix lists and writes:

```text
real_matrices_info.csv
synthetic_matrices_info.csv
```

## Graphing
The script ``graphing/generate.sh`` generates the various graphs used in the report.

Its first parameter must be the path to the root folder containing part2 with the results computed on disk.

Its second parameter must be the path to the merged CPU baseline csv.

Install the dependencies listed in the ``requirements.txt`` file in a virtual environment before running the generation script.

## Notes

The MPI benchmark uses the matrix lists in `matrices.sh`.

The CPU baseline workflow uses `baseline_matrices.sh` inside `multiGPU-SpMV`.
