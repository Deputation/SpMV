#include "benchmark_results.h"
#include "benchmark_values.h"
#include "cpu.h"
#include "csr.cuh"
#include "gpu_utils.cuh"
#include "host_utils.cuh"
#include "constants.h"

#include <cstdint>
#include <cusparse.h>

benchmark_results
benchmark_csr_cpu(const int64_t *host_csr_arow, const int64_t *host_acol,
                  const valuetype *host_aval, const int64_t rows,
                  const int64_t cols, const int64_t nnz,
                  valuetype **ptr_to_dev_out_csr, const valuetype *dev_dense_vec) {
  valuetype *host_out_csr = (valuetype *)malloc(sizeof(valuetype) * rows);
  assert(host_out_csr != nullptr);
  memset(host_out_csr, 0, sizeof(valuetype) * rows);

  valuetype *host_dense_vec = (valuetype *)malloc(sizeof(valuetype) * cols);
  assert(host_dense_vec != nullptr);
  zero_check(cudaMemcpy(host_dense_vec, dev_dense_vec, sizeof(valuetype) * cols,
                        cudaMemcpyDeviceToHost));

  double cpu_time[NITER];

  for (size_t i = 0; i < WARMUP; i++) {
    memset(host_out_csr, 0, sizeof(valuetype) * rows);

    spmv_csr(host_csr_arow, host_acol, host_aval, rows, host_dense_vec,
             host_out_csr);
  }

  for (size_t i = 0; i < NITER; i++) {
    memset(host_out_csr, 0, sizeof(valuetype) * rows);

    TIMER_DEF(CpuTime);

    TIMER_START(CpuTime);

    spmv_csr(host_csr_arow, host_acol, host_aval, rows, host_dense_vec,
             host_out_csr);

    TIMER_STOP(CpuTime);

    cpu_time[i] = TIMER_ELAPSED(CpuTime) / 1e6;
  }

  double mean_time = arithmetic_mean(cpu_time, NITER);
  double geo_mean_time = geometric_mean(cpu_time, NITER);

  zero_check(cudaMalloc(ptr_to_dev_out_csr, sizeof(valuetype) * rows));
  zero_check(cudaMemcpy(*ptr_to_dev_out_csr, host_out_csr,
                        sizeof(valuetype) * rows, cudaMemcpyHostToDevice));

  free(host_out_csr);
  free(host_dense_vec);

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(cpu_time, mean_time, NITER);
  results.error = 0;
  results.gflops = (total_flops / mean_time) / 1e9;

  strcpy(results.method, "CSR_CPU");

  return results;
}

benchmark_results
benchmark_csr_scalar(const int64_t *dev_csr_arow, const int64_t *dev_acol,
                           const valuetype *dev_aval, const int64_t rows,
                           const int64_t nnz, const valuetype *dev_out_csr,
                           const valuetype *dev_dense_vec) {
  valuetype *dev_out_csr_computed = 0;
  zero_check(cudaMalloc(&dev_out_csr_computed, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

  valuetype *dev_out_csr_fake = 0;
  zero_check(cudaMalloc(&dev_out_csr_fake, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_csr_fake, 0, sizeof(valuetype) * rows));

  double gpu_time_math[NITER];

  cudaDeviceProp prop;
  zero_check(cudaGetDeviceProperties(&prop, 0));
  int64_t blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for (size_t i = 0; i < WARMUP; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    spmv_csr_naive<<<blocks, THREADS_PER_BLOCK>>>(dev_csr_arow, dev_acol, dev_aval, rows,
                                                  dev_dense_vec, dev_out_csr_computed);
    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuMathTime);

  for (size_t i = 0; i < NITER; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    CUDA_START_EVENT_TIMER(GpuMathTime);
    spmv_csr_naive<<<blocks, THREADS_PER_BLOCK>>>(dev_csr_arow, dev_acol, dev_aval, rows,
                                                  dev_dense_vec, dev_out_csr_computed);
    CUDA_STOP_EVENT_TIMER(GpuMathTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuMathTime, timer);

    gpu_time_math[i] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuMathTime);

  double gpu_time_bandwidth[NITER];

  for (size_t i = 0; i < WARMUP; i++) {
    // this kernel simulates the write, so we write to an alternative output
    // buffer with the same characteristics
    spmv_csr_naive_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_fake);

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuBwTime);

  for (size_t i = 0; i < NITER; i++) {
    CUDA_START_EVENT_TIMER(GpuBwTime);
    spmv_csr_naive_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_fake);
    CUDA_STOP_EVENT_TIMER(GpuBwTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuBwTime, timer);

    gpu_time_bandwidth[i] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuBwTime);

  double mean_time = arithmetic_mean(gpu_time_math, NITER);
  double fake_kernel_mean_time = arithmetic_mean(gpu_time_bandwidth, NITER);

  double geo_mean_time = geometric_mean(gpu_time_math, NITER);
  double fake_kernel_geo_mean_time = geometric_mean(gpu_time_bandwidth, NITER);

  valuetype total_error = compute_mse_gpu(dev_out_csr_computed, dev_out_csr, rows);

  zero_check(cudaFree(dev_out_csr_computed));
  zero_check(cudaFree(dev_out_csr_fake));

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(gpu_time_math, mean_time, NITER);
  results.error = total_error;
  results.gflops = (total_flops / mean_time) / 1e9;
  results.effective_bw = csr_formula;

  strcpy(results.method, "CSR_SCALAR");

  return results;
}

benchmark_results
benchmark_csr_vector_warp(const int64_t *dev_csr_arow, const int64_t *dev_acol,
                              const valuetype *dev_aval, const int64_t rows,
                              const int64_t nnz, const valuetype *dev_out_csr,
                              const valuetype *dev_dense_vec) {
  valuetype *dev_out_csr_computed = 0;
  zero_check(cudaMalloc(&dev_out_csr_computed, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

  valuetype *dev_out_csr_fake = 0;
  zero_check(cudaMalloc(&dev_out_csr_fake, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_csr_fake, 0, sizeof(valuetype) * rows));

  double gpu_time_math[NITER];
  int64_t blocks =
      ((rows * WARP_SIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for (size_t i = 0; i < WARMUP; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    spmv_csr_warp_row<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_computed);
    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuMathTime);

  for (size_t i = 0; i < NITER; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    CUDA_START_EVENT_TIMER(GpuMathTime);
    spmv_csr_warp_row<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_computed);
    CUDA_STOP_EVENT_TIMER(GpuMathTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuMathTime, timer);

    gpu_time_math[i] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuMathTime);

  double gpu_time_bandwidth[NITER];

  for (size_t i = 0; i < WARMUP; i++) {
    // this kernel simulates the write, so we write to an alternative output
    // buffer with the same characteristics
    spmv_csr_warp_row_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_fake);

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuBwTime);

  for (size_t i = 0; i < NITER; i++) {
    CUDA_START_EVENT_TIMER(GpuBwTime);
    spmv_csr_warp_row_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_csr_arow, dev_acol, dev_aval, rows, dev_dense_vec, dev_out_csr_fake);
    CUDA_STOP_EVENT_TIMER(GpuBwTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuBwTime, timer);

    gpu_time_bandwidth[i] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuBwTime);

  double mean_time = arithmetic_mean(gpu_time_math, NITER);
  double fake_kernel_mean_time = arithmetic_mean(gpu_time_bandwidth, NITER);

  double geo_mean_time = geometric_mean(gpu_time_math, NITER);
  double fake_kernel_geo_mean_time = geometric_mean(gpu_time_bandwidth, NITER);

  valuetype total_error = compute_mse_gpu(dev_out_csr_computed, dev_out_csr, rows);

  zero_check(cudaFree(dev_out_csr_computed));
  zero_check(cudaFree(dev_out_csr_fake));

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(gpu_time_math, mean_time, NITER);
  results.error = total_error;
  results.gflops = (total_flops / mean_time) / 1e9;
  results.effective_bw = csr_warp_row_formula;

  strcpy(results.method, "CSR_VECTOR_WARP");

  return results;
}

benchmark_results benchmark_cusparse_csr_gpu(
    const cusparseHandle_t handle, const int64_t *dev_csr_arow, const int64_t *dev_acol,
    const valuetype *dev_aval, const int64_t rows, const int64_t cols,
    const int64_t nnz, const valuetype *dev_out_csr,
    const valuetype *dev_dense_vec) {
  valuetype *dev_out_csr_computed = 0;
  zero_check(cudaMalloc(&dev_out_csr_computed, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

  double gpu_time[NITER];

  cusparseConstDnVecDescr_t dense_vec_desc;
  zero_check(cusparseCreateConstDnVec(&dense_vec_desc, cols, dev_dense_vec,
                                      cudaDataType::CUDA_R_32F));

  cusparseDnVecDescr_t out_csr_desc;
  zero_check(cusparseCreateDnVec(&out_csr_desc, rows, dev_out_csr_computed,
                                 cudaDataType::CUDA_R_32F));

  cusparseConstSpMatDescr_t sparse_matrix_desc;
  zero_check(cusparseCreateConstCsr(
      &sparse_matrix_desc, rows, cols, nnz, dev_csr_arow, dev_acol, dev_aval,
      cusparseIndexType_t::CUSPARSE_INDEX_64I,
      cusparseIndexType_t::CUSPARSE_INDEX_64I,
      cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType::CUDA_R_32F));

  valuetype alpha = 1.0f;
  valuetype beta = 0.0f;

  size_t buffer;
  zero_check(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse_matrix_desc,
      dense_vec_desc, &beta, out_csr_desc, cudaDataType::CUDA_R_32F,
      CUSPARSE_SPMV_ALG_DEFAULT, &buffer));

  void *buffer_ptr = 0;
  zero_check(cudaMalloc(&buffer_ptr, buffer));

  for (size_t i = 0; i < WARMUP; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            sparse_matrix_desc, dense_vec_desc, &beta,
                            out_csr_desc, cudaDataType::CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuMathTime);

  for (size_t i = 0; i < NITER; i++) {
    zero_check(cudaMemset(dev_out_csr_computed, 0, sizeof(valuetype) * rows));

    CUDA_START_EVENT_TIMER(GpuMathTime);
    zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            sparse_matrix_desc, dense_vec_desc, &beta,
                            out_csr_desc, cudaDataType::CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));
    CUDA_STOP_EVENT_TIMER(GpuMathTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuMathTime, timer);

    gpu_time[i] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuMathTime);

  double mean_time = arithmetic_mean(gpu_time, NITER);
  double geo_mean_time = geometric_mean(gpu_time, NITER);

  valuetype total_error = compute_mse_gpu(dev_out_csr_computed, dev_out_csr, rows);

  zero_check(cudaFree(dev_out_csr_computed));
  zero_check(cudaFree(buffer_ptr));

  zero_check(cusparseDestroyDnVec(dense_vec_desc));
  zero_check(cusparseDestroyDnVec(out_csr_desc));
  zero_check(cusparseDestroySpMat(sparse_matrix_desc));

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(gpu_time, mean_time, NITER);
  results.error = total_error;
  results.gflops = (total_flops / mean_time) / 1e9;

  strcpy(results.method, "CSR_CUSPARSE");

  return results;
}
