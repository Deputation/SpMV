#!/bin/bash

source ./matrices.sh

for matrix in "${real_matrices[@]}"; do
	./output/matrix_info $matrix
done

cp matrix_info.csv real_matrices_info.csv
rm matrix_info.csv

for matrix in "${synthetic_matrices[@]}"; do
	./output/matrix_info $matrix
done

cp matrix_info.csv synthetic_matrices_info.csv
rm matrix_info.csv
