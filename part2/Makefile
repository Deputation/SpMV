NVCC ?= nvcc
MPI_CXX ?= mpicxx

ARCH ?= sm_80
LOCAL ?= 0

OUTPUT_DIR  ?= output
RESULTS_DIR ?= results
LOGS_DIR    ?= logs
MERGED_RESULTS_DIR ?= merged_results

INCLUDE_DIRS = \
	-I src/libraries \
	-I src/libraries/benchmarks \
	-I src/libraries/kernels \
	-I src/libraries/parsing \
	-I src/libraries/utils

ifeq ($(LOCAL), 1)
MPI_COMPILE_FLAGS = -I/usr/include/x86_64-linux-gnu/openmpi
MPI_LINK_FLAGS = -lmpi
else
MPI_COMPILE_FLAGS := $(shell $(MPI_CXX) --showme:compile)
MPI_LINK_FLAGS := $(shell $(MPI_CXX) --showme:link | sed 's/-Wl,/ -Xlinker /g')
endif

WARNINGS = -Wall -Wextra -Wno-unused-function

NVCC_OPTIONS = \
	-g -O2 \
	-D_FILE_OFFSET_BITS=64 \
	--extended-lambda \
	--gpu-architecture=$(ARCH) \
	-Xcompiler="-fopenmp $(WARNINGS)"

CUDA_LIBS = -lm -lcudart -lcurand -lcusparse

PARSING = \
	src/libraries/parsing/parse_file.cu \
	src/libraries/parsing/entry.cpp

PARALLEL_PARSING = \
	src/libraries/parsing/parallel_parse_file.cpp

UTILS = \
	src/libraries/utils/gpu_utils.cu \
	src/libraries/utils/host_utils.cu

KERNELS = \
	src/libraries/kernels/coo.cu \
	src/libraries/kernels/csr.cu \
	src/libraries/kernels/cpu.cpp

MPI_KERNELS = \
	src/libraries/kernels/mpi_gpu_coo.cu \
	src/libraries/kernels/mpi_gpu_csr.cu \
	src/libraries/kernels/mpi_cpu.cpp

MPI_UTILS = \
	src/libraries/utils/host_mpi_utils.cpp \
	src/libraries/utils/gpu_mpi_utils.cu

MPI_BENCHMARKING = \
	src/libraries/benchmarks/mpi_coo.cu \
	src/libraries/benchmarks/mpi_csr.cu \
	src/libraries/benchmarks/mpi_benchmark_results.cu

.PHONY: all dirs clean

all: output/mpi_spmv output/matrix_info output/matrix_generator

dirs:
	mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR) $(LOGS_DIR) $(MERGED_RESULTS_DIR)

output/mpi_spmv: src/programs/mpi_1d.cu | dirs
	$(NVCC) $(NVCC_OPTIONS) $(MPI_COMPILE_FLAGS) $(INCLUDE_DIRS) \
		$(PARSING) $(PARALLEL_PARSING) $(UTILS) $(MPI_UTILS) \
		$(KERNELS) $(MPI_KERNELS) $(MPI_BENCHMARKING) $^ \
		-o $@ $(CUDA_LIBS) $(MPI_LINK_FLAGS)

output/matrix_info: $(PARSING) src/programs/matrix_info.cu | dirs
	$(NVCC) $(NVCC_OPTIONS) $(INCLUDE_DIRS) $^ -o $@ $(CUDA_LIBS)

output/matrix_generator: src/programs/matrix_generator.cpp | dirs
	g++ -O2 -g -Wall -Wextra -Wpedantic -std=c++17 $^ -o $@

clean:
	rm -rf $(OUTPUT_DIR) $(RESULTS_DIR) $(LOGS_DIR) $(MERGED_RESULTS_DIR)
	rm -f *.csv *.out *.err *.log
