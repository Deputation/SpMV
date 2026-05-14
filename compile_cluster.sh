#!/bin/bash

mkdir -p output
mkdir -p results
mkdir -p logs

module load CUDA/12.5.0

make -j$(nproc)

module purge
