#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "constants.h"

#include <cstdint>
#include <cusparse.h>

#define CUDA_EVENT_TIMER_DEF(name)                                             \
  cudaEvent_t start_##name;                                                    \
  cudaEvent_t stop_##name;                                                     \
  cudaEventCreate(&start_##name);                                              \
  cudaEventCreate(&stop_##name);

#define CUDA_START_EVENT_TIMER(name) cudaEventRecord(start_##name, 0);
#define CUDA_STOP_EVENT_TIMER(name)                                            \
  cudaEventRecord(stop_##name, 0);                                             \
  cudaEventSynchronize(stop_##name);
#define CUDA_EVENT_TIMER_ELAPSED(name, variable)                               \
  cudaEventElapsedTime(&variable, start_##name, stop_##name);

#define CUDA_EVENT_TIMER_DESTROY(name)                                         \
  cudaEventDestroy(start_##name);                                              \
  cudaEventDestroy(stop_##name);

#define WARP_SIZE 32

#define cusparse_zero_check(status)                                            \
  if (status != CUSPARSE_STATUS_SUCCESS) {                                     \
    fprintf(stderr, "cuSPARSE error: %d\n", static_cast<int>(status));         \
    exit(1);                                                                   \
  }

static cusparseIndexType_t cusparse_idxtype() {
  if (sizeof(idxtype) == 4) {
    return CUSPARSE_INDEX_32I;
  }

  if (sizeof(idxtype) == 8) {
    return CUSPARSE_INDEX_64I;
  }

  printf("unsupported idxtype size for cuSPARSE\n");
  exit(1);
}

__global__ void compute_error_gpu(const vtype *vec_one,
                                  const vtype *vec_two, double *error,
                                  size_t entries);

double compute_mse_gpu(const vtype *dev_vec_one,
                       const vtype *dev_vec_two, const size_t entries);

vtype *init_dense_vec_dev(idxtype cols);

#endif
