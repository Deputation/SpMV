# SpMV on GPU Investigation (Deliverable #1)

This repository contains the source code, build scripts, and execution environments for evaluating Sparse Matrix-Vector Multiplication (SpMV) kernels using COO and CSR storage formats. The benchmarks compare custom unoptimized and an optimized GPU kernel against OpenMP CPU baselines and NVIDIA's cuSPARSE library.

## TLDR
Please make sure the dataset is in place before running the following commands (see **Dataset Setup** section).
### SLURM Cluster
Running on a configured SLURM Cluster using the partitions and nodes defined in the scripts:
```
./compile_cluster.sh
./run_parallel_cluster.sh
sbatch run_info_cluster.sh
# WAIT FOR ALL JOBS TO FINISH, THEN RUN
./merge.sh
```
The results.csv and matrix_info.csv files will be in the root folder.

### Local Machine
Running on a local machine with a NVIDIA GPU.
```
./compile.sh
./run_serial.sh
./run_info.sh
./merge.sh
```
The results.csv and matrix_info.csv files will be in the root folder.

## Prerequisites

To compile and run this project, the following dependencies are required:
* CUDA Toolkit (Tested with version 12.5.0)
* OpenMP
* Valgrind (Specifically the `cachegrind` tool for cache profiling)
* C++ Compiler compatible with `nvcc`

## Dataset Setup

The dataset is **not** included in this repository. You must download the matrices from the SuiteSparse Matrix Collection.

Create a directory named `dataset` in the root of the repository and extract the `.tar.gz` files directly into it. The parsing scripts expect a nested folder structure where each matrix has its own directory.

**Expected Structure:**
```text
SpMV/
├── dataset/
│   ├── ASIC_680ks/
│   │   └── ASIC_680ks.mtx
│   ├── FullChip/
│   │   └── FullChip.mtx
│   └── ... (other matrices)
```

## Compilation

Utility scripts are provided to create the necessary output directories (`output/`, `results/`, `logs/`) and trigger the `Makefile`.

* For Local Machines: Run `./compile.sh`
* For SLURM Clusters: Run `./compile_cluster.sh` (This automatically loads the `CUDA/12.5.0` module before compiling, then purges the loaded modules).

## Execution: SLURM Cluster Workflow

The cluster scripts are pre-configured with SLURM directives (requesting the `edu-short` partition, `gpu.computing26` account, and `edu01` nodelist).

**Running the Benchmarks**: Run `./run_parallel_cluster.sh`. This script acts as a job manager and will automatically queue separate `sbatch` jobs for every matrix benchmark and cache study.

**Computing Matrix Structural Info**: Run `sbatch run_info_cluster.sh`. This submits a job to parse the dataset and compute sparsity, Max/Avg, and CV metrics.

**Hardware Info**: Run `sbatch cpu_info.sh` to dump the cluster node's CPU specifications to the logs.

## Execution: Local Machine Workflow

If you are running this on a local workstation rather than a cluster, use the standard shell scripts:

**Benchmarks**: Run `./run_serial.sh` to execute the COO and CSR evaluations across the dataset sequentially.

**Matrix Info**: Run `./run_info.sh` to compute the structural properties of the matrices.

**Cache Profiling**: Run `./cache.sh` to execute the CPU baselines through Valgrind's `cachegrind` tool (currently targeted at `Ga41As41H72.mtx`).

## Aggregating Results

The benchmarking scripts output individual `.csv` files into the `results/` directory for each benchmarked kernel. To view the final consolidated data:

**Benchmark Results**: You **must** run `./merge.sh` after your benchmarks finish. This script concatenates all the individual outputs from the `results/` folder and generates a single, clean `results.csv` file in the root directory. This file was used to generate all graphs in the report and contains all relevant information used to compare the algorithms.

**Cache Study**: Cache results can be found in the ``logs`` folder, both in the form of summaries and as parsable ``.out`` files.

**Matrix Information**: The structural metrics extracted by the info executable are automatically saved directly to `matrix_info.csv` in the root directory.
