#!/bin/bash

make clean
LOCAL=1 make -j$(nproc)
