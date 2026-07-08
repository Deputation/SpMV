CC=nvcc
COMMON_OPTIONS=-g -Xcompiler "-fopenmp -Wall -Wextra -Wno-unused-function" --extended-lambda -O2 --gpu-architecture=sm_80 -lm -lcusparse -lcudart -lcurand

INCLUDE_DIRS=-I src/libraries -I src/libraries/benchmarks -I src/libraries/kernels -I src/libraries/parsing -I src/libraries/utils

KERNELS=src/libraries/kernels/cpu.cpp  src/libraries/kernels/coo.cu src/libraries/kernels/csr.cu
UTILS=src/libraries/utils/gpu_utils.cu src/libraries/utils/host_utils.cu
BENCHMARKS=src/libraries/benchmarks/coo_benchmark.cu src/libraries/benchmarks/csr_benchmark.cu src/libraries/benchmarks/benchmark_results.cpp
PARSING=src/libraries/parsing/parse_file.cu src/libraries/parsing/entry.cpp

COMMON=$(KERNELS) $(UTILS) $(BENCHMARKS) $(PARSING)

all: spmv_coo spmv_csr spmv_cpu_coo spmv_cpu_csr matrix_info

spmv_coo: $(COMMON) src/programs/coo.cu
	$(CC) $(COMMON_OPTIONS) $^ -o output/$@ $(INCLUDE_DIRS)

spmv_csr: $(COMMON) src/programs/csr.cu
	$(CC) $(COMMON_OPTIONS) $^ -o output/$@ $(INCLUDE_DIRS)

spmv_cpu_coo: $(COMMON) src/programs/cpu_coo.cu
	$(CC) $(COMMON_OPTIONS) $^ -o output/$@ $(INCLUDE_DIRS)

spmv_cpu_csr: $(COMMON) src/programs/cpu_csr.cu
	$(CC) $(COMMON_OPTIONS) $^ -o output/$@ $(INCLUDE_DIRS)

matrix_info: $(COMMON) src/programs/matrix_info.cu
	$(CC) $(COMMON_OPTIONS) $^ -o output/$@ $(INCLUDE_DIRS)

clean:
	rm output/spmv_coo output/spmv_csr output/spmv_cpu_coo output/spmv_cpu_csr output/matrix_info
