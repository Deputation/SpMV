#!/bin/bash

mkdir -p output
mkdir -p results
mkdir -p logs

make -j$(nproc)
