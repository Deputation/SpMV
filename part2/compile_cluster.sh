#!/bin/bash

make clean
module purge
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

make -j$(nproc)

module purge
