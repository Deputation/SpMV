#!/bin/bash

mkdir -p synthetic_dataset

./output/matrix_generator 1000000 1000000 10000000 real_general 1 42 synthetic_dataset/synthetic-10M.mtx
./output/matrix_generator 1000000 1000000 10000000 real_general 1000 42 synthetic_dataset/synthetic-10M-unbalanced.mtx

./output/matrix_generator 1000000 1000000 20000000 real_general 1 42 synthetic_dataset/synthetic-20M.mtx
./output/matrix_generator 1000000 1000000 20000000 real_general 1000 42 synthetic_dataset/synthetic-20M-unbalanced.mtx

./output/matrix_generator 1000000 1000000 40000000 real_general 1 42 synthetic_dataset/synthetic-40M.mtx
./output/matrix_generator 1000000 1000000 40000000 real_general 1000 42 synthetic_dataset/synthetic-40M-unbalanced.mtx
