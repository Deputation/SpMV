#include "benchmark_results.h"
#include "benchmark_values.h"
#include "coo_benchmark.cuh"
#include "cpu.h"
#include "coo.cuh"
#include "gpu_utils.cuh"
#include "host_utils.cuh"
#include "constants.h"

#include <cstdint>
#include <cusparse.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

benchmark_results benchmark_coo_cpu(const int64_t *host_arow,
                                    const int64_t *host_acol,
                                    const valuetype *host_aval,
                                    const int64_t rows, const int64_t cols,
                                    const int64_t nnz,
                                    valuetype **ptr_to_dev_out_coo,
                                    const valuetype *dev_dense_vec) {
  valuetype *host_out_coo = (valuetype *)malloc(sizeof(valuetype) * rows);
  assert(host_out_coo != nullptr);
  memset(host_out_coo, 0, sizeof(valuetype) * rows);

  valuetype *host_dense_vec = (valuetype *)malloc(sizeof(valuetype) * cols);
  assert(host_dense_vec != nullptr);
  zero_check(cudaMemcpy(host_dense_vec, dev_dense_vec, sizeof(valuetype) * cols,
                        cudaMemcpyDeviceToHost));

  double cpu_time[NITER];

  for (size_t i = 0; i < WARMUP; i++) {
    memset(host_out_coo, 0, sizeof(valuetype) * rows);

    spmv_coo(host_arow, host_acol, host_aval, nnz, host_dense_vec,
             host_out_coo);
  }

  for (size_t i = 0; i < NITER; i++) {
    memset(host_out_coo, 0, sizeof(valuetype) * rows);

    TIMER_DEF(CpuTime);

    TIMER_START(CpuTime);

    spmv_coo(host_arow, host_acol, host_aval, nnz, host_dense_vec,
             host_out_coo);

    TIMER_STOP(CpuTime);

    cpu_time[i] = TIMER_ELAPSED(CpuTime) / 1e6;
  }

  double mean_time = arithmetic_mean(cpu_time, NITER);
  double geo_mean_time = geometric_mean(cpu_time, NITER);

  zero_check(cudaMalloc(ptr_to_dev_out_coo, sizeof(valuetype) * rows));
  zero_check(cudaMemcpy(*ptr_to_dev_out_coo, host_out_coo,
                        sizeof(valuetype) * rows, cudaMemcpyHostToDevice));

  free(host_dense_vec);
  free(host_out_coo);

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(cpu_time, mean_time, NITER);
  results.error = 0;
  results.gflops = (total_flops / mean_time) / 1e9;

  strcpy(results.method, "COO_CPU");

  return results;
}

benchmark_results
benchmark_coo_kernel_naive(const int64_t *dev_arow, const int64_t *dev_acol,
                           const valuetype *dev_aval, const int64_t rows,
                           const int64_t nnz, const valuetype *dev_out_coo,
                           const valuetype *dev_dense_vec) {
  valuetype *dev_out_coo_computed = 0;
  zero_check(cudaMalloc(&dev_out_coo_computed, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

  valuetype *dev_out_coo_fake = 0;
  zero_check(cudaMalloc(&dev_out_coo_fake, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_coo_fake, 0, sizeof(valuetype) * rows));

  double gpu_time_math[NITER];

  cudaDeviceProp prop;
  zero_check(cudaGetDeviceProperties(&prop, 0));
  int64_t blocks = (nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for (size_t i = 0; i < WARMUP; i++) {
    zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

    spmv_coo_naive<<<blocks, THREADS_PER_BLOCK>>>(dev_arow, dev_acol, dev_aval,
                                                  nnz, rows, dev_dense_vec,
                                                  dev_out_coo_computed);
    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuMathTime);

  for (size_t i = 0; i < NITER; i++) {
    zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

    CUDA_START_EVENT_TIMER(GpuMathTime);
    spmv_coo_naive<<<blocks, THREADS_PER_BLOCK>>>(dev_arow, dev_acol, dev_aval,
                                                  nnz, rows, dev_dense_vec,
                                                  dev_out_coo_computed);
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
    spmv_coo_naive_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_arow, dev_acol, dev_aval, nnz, rows, dev_dense_vec,
        dev_out_coo_fake);

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuBwTime);

  for (size_t i = 0; i < NITER; i++) {
    CUDA_START_EVENT_TIMER(GpuBwTime);
    spmv_coo_naive_fake<<<blocks, THREADS_PER_BLOCK>>>(
        dev_arow, dev_acol, dev_aval, nnz, rows, dev_dense_vec,
        dev_out_coo_fake);

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

  valuetype total_error =
      compute_mse_gpu(dev_out_coo_computed, dev_out_coo, rows);

  zero_check(cudaFree(dev_out_coo_computed));
  zero_check(cudaFree(dev_out_coo_fake));

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(gpu_time_math, mean_time, NITER);
  results.error = total_error;
  results.gflops = (total_flops / mean_time) / 1e9;
  results.effective_bw = coo_formula;

  strcpy(results.method, "COO_NAIVE");

  return results;
}

void sort_dev_in_place(int64_t *dev_row, int64_t *dev_col, valuetype *dev_aval,
                       size_t nnz) {
  auto start =
      thrust::make_zip_iterator(thrust::make_tuple(dev_row, dev_col, dev_aval));
  auto end = start + nnz;

  thrust::sort(
      thrust::device, start, end,
      [] __device__(
          const thrust::tuple<int64_t, int64_t, valuetype> &a,
          const thrust::tuple<int64_t, int64_t, valuetype> &b) -> const bool {
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
          return thrust::get<0>(a) < thrust::get<0>(b);
        }
        return thrust::get<1>(a) < thrust::get<1>(b);
      });
}

benchmark_results benchmark_cusparse_coo_gpu(
    const cusparseHandle_t handle, const int64_t *dev_arow,
    const int64_t *dev_acol, const valuetype *dev_aval, const int64_t rows,
    const int64_t cols, const int64_t nnz, const valuetype *dev_out_coo,
    const valuetype *dev_dense_vec) {
  valuetype *dev_out_coo_computed = 0;
  zero_check(cudaMalloc(&dev_out_coo_computed, sizeof(valuetype) * rows));
  zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

  // copy coo values
  int64_t *dev_arow_copy = 0;
  zero_check(cudaMalloc(&dev_arow_copy, sizeof(int64_t) * nnz));

  int64_t *dev_acol_copy = 0;
  zero_check(cudaMalloc(&dev_acol_copy, sizeof(int64_t) * nnz));

  valuetype *dev_aval_copy = 0;
  zero_check(cudaMalloc(&dev_aval_copy, sizeof(valuetype) * nnz));

  double sorting_time[NITER];

  for (int warmup = 0; warmup < WARMUP; warmup++) {
    zero_check(cudaMemcpy(dev_arow_copy, dev_arow, sizeof(int64_t) * nnz,
                          cudaMemcpyDefault));
    zero_check(cudaMemcpy(dev_acol_copy, dev_acol, sizeof(int64_t) * nnz,
                          cudaMemcpyDefault));
    zero_check(cudaMemcpy(dev_aval_copy, dev_aval, sizeof(valuetype) * nnz,
                          cudaMemcpyDefault));

    sort_dev_in_place(dev_arow_copy, dev_acol_copy, dev_aval_copy, nnz);

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuSortTime);

  for (int iter = 0; iter < NITER; iter++) {
    zero_check(cudaMemcpy(dev_arow_copy, dev_arow, sizeof(int64_t) * nnz,
                          cudaMemcpyDefault));
    zero_check(cudaMemcpy(dev_acol_copy, dev_acol, sizeof(int64_t) * nnz,
                          cudaMemcpyDefault));
    zero_check(cudaMemcpy(dev_aval_copy, dev_aval, sizeof(valuetype) * nnz,
                          cudaMemcpyDefault));

    CUDA_START_EVENT_TIMER(GpuSortTime);

    sort_dev_in_place(dev_arow_copy, dev_acol_copy, dev_aval_copy, nnz);

    CUDA_STOP_EVENT_TIMER(GpuSortTime);

    float timer = 0.0f;
    CUDA_EVENT_TIMER_ELAPSED(GpuSortTime, timer);

    sorting_time[iter] = timer / 1e3;

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DESTROY(GpuSortTime);

  double gpu_time[NITER];

  cusparseConstDnVecDescr_t dense_vec_desc;
  zero_check(cusparseCreateConstDnVec(&dense_vec_desc, cols, dev_dense_vec,
                                      cudaDataType::CUDA_R_32F));

  cusparseDnVecDescr_t out_coo_desc;
  zero_check(cusparseCreateDnVec(&out_coo_desc, rows, dev_out_coo_computed,
                                 cudaDataType::CUDA_R_32F));

  cusparseConstSpMatDescr_t sparse_matrix_desc;
  zero_check(cusparseCreateConstCoo(
      &sparse_matrix_desc, rows, cols, nnz, dev_arow_copy, dev_acol_copy,
      dev_aval_copy, cusparseIndexType_t::CUSPARSE_INDEX_64I,
      cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType::CUDA_R_32F));

  valuetype alpha = 1.0f;
  valuetype beta = 0.0f;

  size_t buffer;
  zero_check(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse_matrix_desc,
      dense_vec_desc, &beta, out_coo_desc, cudaDataType::CUDA_R_32F,
      CUSPARSE_SPMV_COO_ALG1, &buffer));

  void *buffer_ptr = 0;
  zero_check(cudaMalloc(&buffer_ptr, buffer));

  for (size_t i = 0; i < WARMUP; i++) {
    zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

    zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            sparse_matrix_desc, dense_vec_desc, &beta,
                            out_coo_desc, cudaDataType::CUDA_R_32F,
                            CUSPARSE_SPMV_COO_ALG1, buffer_ptr));

    zero_check(cudaDeviceSynchronize());
    zero_check(cudaGetLastError());
  }

  CUDA_EVENT_TIMER_DEF(GpuMathTime);

  for (size_t i = 0; i < NITER; i++) {
    zero_check(cudaMemset(dev_out_coo_computed, 0, sizeof(valuetype) * rows));

    CUDA_START_EVENT_TIMER(GpuMathTime);
    zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            sparse_matrix_desc, dense_vec_desc, &beta,
                            out_coo_desc, cudaDataType::CUDA_R_32F,
                            CUSPARSE_SPMV_COO_ALG1, buffer_ptr));

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

  double sorting_mean_time = arithmetic_mean(sorting_time, NITER);
  double sorting_geo_mean_time = geometric_mean(sorting_time, NITER);

  valuetype total_error =
      compute_mse_gpu(dev_out_coo_computed, dev_out_coo, rows);

  zero_check(cudaFree(dev_out_coo_computed));
  zero_check(cudaFree(buffer_ptr));
  zero_check(cudaFree(dev_acol_copy));
  zero_check(cudaFree(dev_arow_copy));
  zero_check(cudaFree(dev_aval_copy));

  zero_check(cusparseDestroyDnVec(dense_vec_desc));
  zero_check(cusparseDestroyDnVec(out_coo_desc));
  zero_check(cusparseDestroySpMat(sparse_matrix_desc));

  int64_t total_flops = nnz * 2;

  benchmark_results results;
  results.time_average = mean_time;
  results.time_average_geo = geo_mean_time;
  results.time_deviation = compute_deviation(gpu_time, mean_time, NITER);
  results.error = total_error;
  results.gflops = (total_flops / mean_time) / 1e9;
  results.preprocessing_time_average = sorting_mean_time;
  results.preprocessing_time_average_geo = sorting_geo_mean_time;
  results.preprocessing_time_deviation =
      compute_deviation(sorting_time, sorting_mean_time, NITER);

  strcpy(results.method, "COO_CUSPARSE");

  return results;
}
