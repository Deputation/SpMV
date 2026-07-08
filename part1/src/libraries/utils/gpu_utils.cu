#include "gpu_utils.cuh"
#include "host_utils.cuh"

#include <cub/cub.cuh>
#include <curand.h>

__global__ void compute_error_gpu(const valuetype *vec_one,
                                  const valuetype *vec_two, double *error,
                                  size_t entries) {
  using WarpReduce = cub::WarpReduce<double>;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  int64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  double sum = 0.0;
  for (int i = idx; i < entries; i += stride) {
    double diff = vec_one[i] - vec_two[i];
    sum += diff * diff;
  }

  double warp_sum = WarpReduce(temp_storage).Sum(sum);

  if ((threadIdx.x % 32) == 0) {
    atomicAdd(error, warp_sum);
  }
}

double compute_mse_gpu(const valuetype *dev_vec_one,
                       const valuetype *dev_vec_two, const size_t entries) {
  int64_t blocks = (entries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  double *dev_sum = 0;
  zero_check(cudaMalloc(&dev_sum, sizeof(double)));
  zero_check(cudaMemset(dev_sum, 0, sizeof(double)));

  compute_error_gpu<<<blocks, THREADS_PER_BLOCK>>>(dev_vec_one, dev_vec_two,
                                                   dev_sum, entries);

  double host_sum = 0;
  zero_check(
      cudaMemcpy(&host_sum, dev_sum, sizeof(double), cudaMemcpyDeviceToHost));
  host_sum /= static_cast<double>(entries);

  zero_check(cudaFree(dev_sum));

  return host_sum;
}

valuetype *init_dense_vec_dev(int64_t cols) {
  valuetype *dev_dense_vec = 0;
  zero_check(cudaMalloc(&dev_dense_vec, sizeof(valuetype) * cols));

  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(generator, 42ull);
  curandGenerateUniform(generator, dev_dense_vec, cols);

  zero_check(cudaDeviceSynchronize());
  zero_check(cudaGetLastError());

  curandDestroyGenerator(generator);

  return dev_dense_vec;
}
