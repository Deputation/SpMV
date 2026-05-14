#!/bin/bash

MATRIX="dataset/Ga41As41H72/Ga41As41H72.mtx"
NAME=$(basename "$MATRIX" .mtx)

valgrind --tool=cachegrind --cache-sim=yes \
         --log-file="logs/cache_coo_${NAME}_%p.txt" \
         --cachegrind-out-file="logs/cache_out_coo_${NAME}_%p.out" \
         ./output/spmv_cpu_coo "$MATRIX"

valgrind --tool=cachegrind --cache-sim=yes \
         --log-file="logs/cache_csr_${NAME}_%p.txt" \
         --cachegrind-out-file="logs/cache_out_csr_${NAME}_%p.out" \
         ./output/spmv_cpu_csr "$MATRIX"
