#ifndef GPU_MACROS_H
#define GPU_MACROS_H

#include "constants.h"

#include <cstdint>

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

__global__ void compute_error_gpu(const valuetype *vec_one,
                                  const valuetype *vec_two, double *error,
                                  size_t entries);

double compute_mse_gpu(const valuetype *dev_vec_one,
                       const valuetype *dev_vec_two, const size_t entries);

valuetype *init_dense_vec_dev(int64_t cols);

#endif
