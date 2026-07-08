#!/bin/bash

source ./matrices.sh

for matrix in "${real_matrices[@]}"; do
	echo "benchmarking $matrix"

	mpirun -n 1 output/mpi_spmv $matrix
	mpirun -n 2 output/mpi_spmv $matrix
	mpirun -n 4 output/mpi_spmv $matrix
done

for matrix in "${synthetic_matrices[@]}"; do
	echo "benchmarking $matrix"

	mpirun -n 1 output/mpi_spmv $matrix
	mpirun -n 2 output/mpi_spmv $matrix
	mpirun -n 4 output/mpi_spmv $matrix
done
