#include "benchmark_values.h"
#include "cpu.h"
#include "gpu_mpi_utils.cuh"
#include "gpu_utils.cuh"
#include "host_mpi_utils.h"
#include "host_utils.cuh"
#include "mpi_benchmark_results.cuh"
#include "mpi_cpu.h"
#include "mpi_csr.cuh"
#include "mpi_gpu_csr.cuh"
#include "parallel_parse_file.h"
#include "parse_file.cuh"
#include <mpi.h>
#include <string.h>

vtype *compute_local_csr_result(const char *file_name, vtype *dense_vec) {
  idxtype *acol, *csr_arow;
  vtype *aval;
  idxtype rows, cols, nnz;
  if (parse_file_csr(file_name, &csr_arow, &acol, &aval, &rows, &cols, &nnz) !=
      0) {
    printf("error encountered when parsing file\n");
    exit(2);
  }

  vtype *out_csr = (vtype *)malloc(sizeof(vtype) * rows);
  assert(out_csr != nullptr);
  memset(out_csr, 0, sizeof(vtype) * rows);
  spmv_csr(csr_arow, acol, aval, rows, dense_vec, out_csr);

  free(csr_arow);
  free(acol);
  free(aval);

  return out_csr;
}

mpi_compute_benchmark_results compute_csr_cpu_and_validate(
    const char *file_path, header_info *header, const idxtype *csr_lrow,
    const idxtype *global_acol, const vtype *aval, idxtype rank_nnz,
    vtype **local_result_out, const vtype *rank_0_dense_vec, int rank,
    int comm_size) {
  mpi_compute_benchmark_results results("CSR_CPU");

  int error = MPI_SUCCESS;

  int local_rows = compute_cyclic_count(header->rows, rank, comm_size);

  assert(csr_lrow[0] == 0);
  assert(csr_lrow[local_rows] == rank_nnz);

  idxtype *editable_acol =
      reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * rank_nnz));
  assert(editable_acol != nullptr);
  memcpy(editable_acol, global_acol, sizeof(idxtype) * rank_nnz);

  vtype *dense_vec = nullptr;
  vtype *dense_vec_cyclic = nullptr;

  if (rank == 0) {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memcpy(dense_vec, rank_0_dense_vec, sizeof(vtype) * header->cols);

    dense_vec_cyclic =
        reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec_cyclic != nullptr);
    memset(dense_vec_cyclic, 0, sizeof(vtype) * header->cols);
  } else {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memset(dense_vec, 0, sizeof(vtype) * header->cols);
  }

  int dense_vec_counts[comm_size];
  int dense_vec_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    dense_vec_counts[i] = compute_cyclic_count(header->cols, i, comm_size);
  }

  int local_dense_vec_total = compute_displacements(
      dense_vec_counts, dense_vec_displacements, comm_size);
  assert(local_dense_vec_total == header->cols);

  if (rank == 0) {
    global_to_cyclic(dense_vec_cyclic, dense_vec, dense_vec_displacements,
                     dense_vec_counts, comm_size);
  }

  int local_dense_vec_count = dense_vec_counts[rank];
  int local_dense_vec_alloc =
      local_dense_vec_count > 0 ? local_dense_vec_count : 1;

  vtype *local_dense_vec =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_dense_vec_alloc));
  assert(local_dense_vec != nullptr);
  memset(local_dense_vec, 0, sizeof(vtype) * local_dense_vec_alloc);

  error = MPI_Scatterv(dense_vec_cyclic, dense_vec_counts,
                       dense_vec_displacements, MPI_FLOAT, local_dense_vec,
                       local_dense_vec_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  if (rank == 0) {
    free(dense_vec_cyclic);
  }

  vtype *extended_dense_vec = nullptr;
  int extended_dense_vec_count = 0;
  int received_columns_count = 0;

  benchmark_exchange_columns(
      global_acol, editable_acol, rank_nnz, header->cols, local_dense_vec,
      dense_vec_counts[rank], rank, comm_size, &extended_dense_vec,
      &extended_dense_vec_count, &received_columns_count, &results);

  vtype *rank_0_result_cyclic = nullptr;
  vtype *rank_0_result = nullptr;

  if (rank == 0) {
    rank_0_result_cyclic =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result_cyclic != nullptr);
    memset(rank_0_result_cyclic, 0, header->rows * sizeof(vtype));

    rank_0_result =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result != nullptr);
    memset(rank_0_result, 0, header->rows * sizeof(vtype));
  }

  int local_rows_alloc = local_rows > 0 ? local_rows : 1;

  vtype *local_result =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_rows_alloc));
  assert(local_result != nullptr);
  memset(local_result, 0, sizeof(vtype) * local_rows);

  double kernel_times[NITER];
  for (size_t i = 0; i < NITER; i++) {
    kernel_times[i] = 0.0;
  }
  double first_run_time = 0.0;

  if (rank_nnz > 0) {
    for (int warmup = 0; warmup < WARMUP; warmup++) {
      // reset result
      memset(local_result, 0, sizeof(vtype) * local_rows);
      spmv_csr_local_out_extended(csr_lrow, editable_acol, aval, local_rows,
                                  extended_dense_vec, local_result);
    }

    for (int iter = 0; iter < NITER; iter++) {
      // reset result
      memset(local_result, 0, sizeof(vtype) * local_rows);

      TIMER_DEF(KernelTime);

      TIMER_START(KernelTime);

      spmv_csr_local_out_extended(csr_lrow, editable_acol, aval, local_rows,
                                  extended_dense_vec, local_result);
      TIMER_STOP(KernelTime);

      kernel_times[iter] = TIMER_ELAPSED(KernelTime) / 1e6;
      if (iter == 0) {
        // we will also record min, max, avg kernel run time for the post warm
        // up run
        first_run_time = kernel_times[iter];
      }
    }
  }

  results.populate_kernel_times(kernel_times, NITER, first_run_time, rank,
                                comm_size);
  results.populate_flop_metrics(rank_nnz, rank, comm_size);

  free(local_dense_vec);
  free(extended_dense_vec);

  free(editable_acol);

  int row_counts[comm_size];
  int row_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    row_counts[i] = compute_cyclic_count(header->rows, i, comm_size);
  }

  int total_rows =
      compute_displacements(row_counts, row_displacements, comm_size);
  assert(total_rows == header->rows);

  error =
      MPI_Gatherv(local_result, local_rows, MPI_FLOAT, rank_0_result_cyclic,
                  row_counts, row_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  free(local_result);

  if (rank == 0) {
    cyclic_to_global(rank_0_result, rank_0_result_cyclic, row_displacements,
                     row_counts, comm_size);

    vtype *local_result = compute_local_csr_result(file_path, dense_vec);
    *local_result_out = local_result;

    double error = compute_mse(rank_0_result, local_result, header->rows);
    printf("error between computations (CSR): %10f\n", error);
    results.error = error;

    free(rank_0_result);
    free(rank_0_result_cyclic);
    // free(local_result);
  }

  free(dense_vec);

  return results;
}

mpi_compute_benchmark_results compute_csr_gpu_and_validate(
    header_info *header, const idxtype *csr_lrow, const idxtype *global_acol,
    const vtype *aval, idxtype rank_nnz, const vtype *local_cpu_result,
    const vtype *rank_0_dense_vec, int rank, int comm_size) {
  mpi_compute_benchmark_results results("CSR_GPU");

  int error = MPI_SUCCESS;

  int local_rows = compute_cyclic_count(header->rows, rank, comm_size);

  assert(csr_lrow[0] == 0);
  assert(csr_lrow[local_rows] == rank_nnz);

  vtype *dense_vec = nullptr;
  vtype *dense_vec_cyclic = nullptr;

  if (rank == 0) {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memcpy(dense_vec, rank_0_dense_vec, sizeof(vtype) * header->cols);

    dense_vec_cyclic =
        reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec_cyclic != nullptr);
    memset(dense_vec_cyclic, 0, sizeof(vtype) * header->cols);
  } else {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memset(dense_vec, 0, sizeof(vtype) * header->cols);
  }

  int dense_vec_counts[comm_size];
  int dense_vec_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    dense_vec_counts[i] = compute_cyclic_count(header->cols, i, comm_size);
  }

  int local_dense_vec_total = compute_displacements(
      dense_vec_counts, dense_vec_displacements, comm_size);
  assert(local_dense_vec_total == header->cols);

  if (rank == 0) {
    global_to_cyclic(dense_vec_cyclic, dense_vec, dense_vec_displacements,
                     dense_vec_counts, comm_size);
  }

  int local_dense_vec_count = dense_vec_counts[rank];
  int local_dense_vec_alloc =
      local_dense_vec_count > 0 ? local_dense_vec_count : 1;

  vtype *local_dense_vec =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_dense_vec_alloc));
  assert(local_dense_vec != nullptr);
  memset(local_dense_vec, 0, sizeof(vtype) * local_dense_vec_alloc);

  error = MPI_Scatterv(dense_vec_cyclic, dense_vec_counts,
                       dense_vec_displacements, MPI_FLOAT, local_dense_vec,
                       local_dense_vec_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  if (rank == 0) {
    free(dense_vec_cyclic);
  }

  vtype *d_local_dense_vec = nullptr;
  zero_check(
      cudaMalloc(&d_local_dense_vec, sizeof(vtype) * local_dense_vec_alloc));
  zero_check(cudaGetLastError());

  if (local_dense_vec_count > 0) {
    zero_check(cudaMemcpy(d_local_dense_vec, local_dense_vec,
                          sizeof(vtype) * local_dense_vec_count,
                          cudaMemcpyHostToDevice));
  }

  idxtype rank_nnz_alloc = rank_nnz > 0 ? rank_nnz : 1;

  idxtype *d_global_acol = nullptr;
  idxtype *d_editable_acol = nullptr;

  zero_check(cudaMalloc(&d_global_acol, sizeof(idxtype) * rank_nnz_alloc));
  zero_check(cudaGetLastError());

  zero_check(cudaMalloc(&d_editable_acol, sizeof(idxtype) * rank_nnz_alloc));
  zero_check(cudaGetLastError());

  if (rank_nnz > 0) {
    zero_check(cudaMemcpy(d_global_acol, global_acol,
                          sizeof(idxtype) * rank_nnz, cudaMemcpyHostToDevice));

    zero_check(cudaMemcpy(d_editable_acol, d_global_acol,
                          sizeof(idxtype) * rank_nnz,
                          cudaMemcpyDeviceToDevice));
  }

  vtype *d_extended_dense_vec = nullptr;
  int extended_dense_vec_count = 0;
  int received_columns_count = 0;

  benchmark_exchange_columns_gpu(
      d_global_acol, d_editable_acol, rank_nnz, header->cols, d_local_dense_vec,
      local_dense_vec_count, rank, comm_size, &d_extended_dense_vec,
      &extended_dense_vec_count, &received_columns_count, &results);

  vtype *rank_0_result_cyclic = nullptr;
  vtype *rank_0_result = nullptr;

  if (rank == 0) {
    rank_0_result_cyclic =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result_cyclic != nullptr);
    memset(rank_0_result_cyclic, 0, header->rows * sizeof(vtype));

    rank_0_result =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result != nullptr);
    memset(rank_0_result, 0, header->rows * sizeof(vtype));
  }

  int local_rows_alloc = local_rows > 0 ? local_rows : 1;

  vtype *local_result =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_rows_alloc));
  assert(local_result != nullptr);
  memset(local_result, 0, sizeof(vtype) * local_rows_alloc);

  double kernel_times[NITER];
  for (size_t i = 0; i < NITER; i++) {
    kernel_times[i] = 0.0;
  }

  double first_run_time = 0.0;

  if (rank_nnz > 0 && local_rows > 0) {
    idxtype *d_csr_lrow = nullptr;
    zero_check(cudaMalloc(&d_csr_lrow, sizeof(idxtype) * (local_rows + 1)));
    zero_check(cudaGetLastError());

    zero_check(cudaMemcpy(d_csr_lrow, csr_lrow,
                          sizeof(idxtype) * (local_rows + 1),
                          cudaMemcpyHostToDevice));

    vtype *d_aval = nullptr;
    zero_check(cudaMalloc(&d_aval, sizeof(vtype) * rank_nnz));
    zero_check(cudaGetLastError());

    zero_check(cudaMemcpy(d_aval, aval, sizeof(vtype) * rank_nnz,
                          cudaMemcpyHostToDevice));

    vtype *d_local_result = nullptr;
    zero_check(cudaMalloc(&d_local_result, sizeof(vtype) * local_rows_alloc));
    zero_check(cudaGetLastError());

    zero_check(cudaMemset(d_local_result, 0, sizeof(vtype) * local_rows_alloc));

    int blocks =
        ((local_rows * WARP_SIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int warmup = 0; warmup < WARMUP; warmup++) {
      zero_check(
          cudaMemset(d_local_result, 0, sizeof(vtype) * local_rows_alloc));

      spmv_csr_local_out_extended_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          d_csr_lrow, d_editable_acol, d_aval, local_rows, d_extended_dense_vec,
          d_local_result);

      zero_check(cudaGetLastError());
      zero_check(cudaDeviceSynchronize());
    }

    CUDA_EVENT_TIMER_DEF(KernelTime);

    for (int iter = 0; iter < NITER; iter++) {
      zero_check(
          cudaMemset(d_local_result, 0, sizeof(vtype) * local_rows_alloc));

      CUDA_START_EVENT_TIMER(KernelTime);

      spmv_csr_local_out_extended_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          d_csr_lrow, d_editable_acol, d_aval, local_rows, d_extended_dense_vec,
          d_local_result);

      CUDA_STOP_EVENT_TIMER(KernelTime);

      float timer = 0.0f;
      CUDA_EVENT_TIMER_ELAPSED(KernelTime, timer);

      kernel_times[iter] = timer / 1e3;

      if (iter == 0) {
        first_run_time = kernel_times[iter];
      }

      zero_check(cudaGetLastError());
      zero_check(cudaDeviceSynchronize());
    }

    CUDA_EVENT_TIMER_DESTROY(KernelTime);

    zero_check(cudaMemcpy(local_result, d_local_result,
                          sizeof(vtype) * local_rows, cudaMemcpyDeviceToHost));

    zero_check(cudaFree(d_csr_lrow));
    zero_check(cudaFree(d_aval));
    zero_check(cudaFree(d_local_result));
  }

  results.populate_kernel_times(kernel_times, NITER, first_run_time, rank,
                                comm_size);
  results.populate_flop_metrics(rank_nnz, rank, comm_size);

  free(local_dense_vec);

  zero_check(cudaFree(d_local_dense_vec));
  zero_check(cudaFree(d_global_acol));
  zero_check(cudaFree(d_editable_acol));

  if (d_extended_dense_vec != nullptr) {
    zero_check(cudaFree(d_extended_dense_vec));
  }

  int row_counts[comm_size];
  int row_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    row_counts[i] = compute_cyclic_count(header->rows, i, comm_size);
  }

  int total_rows =
      compute_displacements(row_counts, row_displacements, comm_size);
  assert(total_rows == header->rows);

  error =
      MPI_Gatherv(local_result, local_rows, MPI_FLOAT, rank_0_result_cyclic,
                  row_counts, row_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  free(local_result);

  if (rank == 0) {
    cyclic_to_global(rank_0_result, rank_0_result_cyclic, row_displacements,
                     row_counts, comm_size);

    double error = compute_mse(rank_0_result, local_cpu_result, header->rows);
    printf("error between computations (CSR, GPU): %10f\n", error);
    results.error = error;

    free(rank_0_result);
    free(rank_0_result_cyclic);
  }

  free(dense_vec);

  return results;
}

mpi_compute_benchmark_results compute_csr_cusparse_and_validate(
    header_info *header, const idxtype *csr_lrow, const idxtype *global_acol,
    const vtype *aval, idxtype rank_nnz, const vtype *local_cpu_result,
    const vtype *rank_0_dense_vec, int rank, int comm_size) {
  mpi_compute_benchmark_results results("CSR_CUSPARSE");

  int error = MPI_SUCCESS;

  int local_rows = compute_cyclic_count(header->rows, rank, comm_size);

  assert(csr_lrow[0] == 0);
  assert(csr_lrow[local_rows] == rank_nnz);

  vtype *dense_vec = nullptr;
  vtype *dense_vec_cyclic = nullptr;

  if (rank == 0) {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memcpy(dense_vec, rank_0_dense_vec, sizeof(vtype) * header->cols);

    dense_vec_cyclic =
        reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec_cyclic != nullptr);
    memset(dense_vec_cyclic, 0, sizeof(vtype) * header->cols);
  } else {
    dense_vec = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * header->cols));
    assert(dense_vec != nullptr);
    memset(dense_vec, 0, sizeof(vtype) * header->cols);
  }

  int dense_vec_counts[comm_size];
  int dense_vec_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    dense_vec_counts[i] = compute_cyclic_count(header->cols, i, comm_size);
  }

  int local_dense_vec_total = compute_displacements(
      dense_vec_counts, dense_vec_displacements, comm_size);
  assert(local_dense_vec_total == header->cols);

  if (rank == 0) {
    global_to_cyclic(dense_vec_cyclic, dense_vec, dense_vec_displacements,
                     dense_vec_counts, comm_size);
  }

  int local_dense_vec_count = dense_vec_counts[rank];
  int local_dense_vec_alloc =
      local_dense_vec_count > 0 ? local_dense_vec_count : 1;

  vtype *local_dense_vec =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_dense_vec_alloc));
  assert(local_dense_vec != nullptr);
  memset(local_dense_vec, 0, sizeof(vtype) * local_dense_vec_alloc);

  error = MPI_Scatterv(dense_vec_cyclic, dense_vec_counts,
                       dense_vec_displacements, MPI_FLOAT, local_dense_vec,
                       local_dense_vec_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  if (rank == 0) {
    free(dense_vec_cyclic);
  }

  vtype *d_local_dense_vec = nullptr;
  zero_check(
      cudaMalloc(&d_local_dense_vec, sizeof(vtype) * local_dense_vec_alloc));
  zero_check(cudaGetLastError());

  if (local_dense_vec_count > 0) {
    zero_check(cudaMemcpy(d_local_dense_vec, local_dense_vec,
                          sizeof(vtype) * local_dense_vec_count,
                          cudaMemcpyHostToDevice));
  }

  idxtype rank_nnz_alloc = rank_nnz > 0 ? rank_nnz : 1;

  idxtype *d_global_acol = nullptr;
  idxtype *d_editable_acol = nullptr;

  zero_check(cudaMalloc(&d_global_acol, sizeof(idxtype) * rank_nnz_alloc));
  zero_check(cudaGetLastError());

  zero_check(cudaMalloc(&d_editable_acol, sizeof(idxtype) * rank_nnz_alloc));
  zero_check(cudaGetLastError());

  if (rank_nnz > 0) {
    zero_check(cudaMemcpy(d_global_acol, global_acol,
                          sizeof(idxtype) * rank_nnz, cudaMemcpyHostToDevice));

    zero_check(cudaMemcpy(d_editable_acol, d_global_acol,
                          sizeof(idxtype) * rank_nnz,
                          cudaMemcpyDeviceToDevice));
  }

  vtype *d_extended_dense_vec = nullptr;
  int extended_dense_vec_count = 0;
  int received_columns_count = 0;

  benchmark_exchange_columns_gpu(
      d_global_acol, d_editable_acol, rank_nnz, header->cols, d_local_dense_vec,
      local_dense_vec_count, rank, comm_size, &d_extended_dense_vec,
      &extended_dense_vec_count, &received_columns_count, &results);

  vtype *rank_0_result_cyclic = nullptr;
  vtype *rank_0_result = nullptr;

  if (rank == 0) {
    rank_0_result_cyclic =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result_cyclic != nullptr);
    memset(rank_0_result_cyclic, 0, header->rows * sizeof(vtype));

    rank_0_result =
        reinterpret_cast<vtype *>(malloc(header->rows * sizeof(vtype)));
    assert(rank_0_result != nullptr);
    memset(rank_0_result, 0, header->rows * sizeof(vtype));
  }

  int local_rows_alloc = local_rows > 0 ? local_rows : 1;

  vtype *local_result =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_rows_alloc));
  assert(local_result != nullptr);
  memset(local_result, 0, sizeof(vtype) * local_rows_alloc);

  double kernel_times[NITER];
  for (size_t i = 0; i < NITER; i++) {
    kernel_times[i] = 0.0;
  }

  double first_run_time = 0.0;

  if (rank_nnz > 0 && local_rows > 0) {
    cudaDataType value_type = cusparse_vtype();

    idxtype *d_csr_lrow = nullptr;
    zero_check(cudaMalloc(&d_csr_lrow, sizeof(idxtype) * (local_rows + 1)));
    zero_check(cudaGetLastError());

    zero_check(cudaMemcpy(d_csr_lrow, csr_lrow,
                          sizeof(idxtype) * (local_rows + 1),
                          cudaMemcpyHostToDevice));

    vtype *d_aval = nullptr;
    zero_check(cudaMalloc(&d_aval, sizeof(vtype) * rank_nnz));
    zero_check(cudaGetLastError());

    zero_check(cudaMemcpy(d_aval, aval, sizeof(vtype) * rank_nnz,
                          cudaMemcpyHostToDevice));

    vtype *d_local_result = nullptr;
    zero_check(cudaMalloc(&d_local_result, sizeof(vtype) * local_rows_alloc));
    zero_check(cudaGetLastError());

    zero_check(cudaMemset(d_local_result, 0, sizeof(vtype) * local_rows_alloc));

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t x = nullptr;
    cusparseDnVecDescr_t y = nullptr;

    cusparse_zero_check(cusparseCreate(&handle));
    cusparse_zero_check(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    cusparse_zero_check(cusparseCreateCsr(
        &mat, static_cast<int64_t>(local_rows),
        static_cast<int64_t>(extended_dense_vec_count),
        static_cast<int64_t>(rank_nnz), d_csr_lrow, d_editable_acol, d_aval,
        cusparse_idxtype(), cusparse_idxtype(), CUSPARSE_INDEX_BASE_ZERO,
        value_type));

    cusparse_zero_check(
        cusparseCreateDnVec(&x, static_cast<int64_t>(extended_dense_vec_count),
                            d_extended_dense_vec, value_type));

    cusparse_zero_check(cusparseCreateDnVec(
        &y, static_cast<int64_t>(local_rows), d_local_result, value_type));

    const vtype alpha = static_cast<vtype>(1.0);
    const vtype beta = static_cast<vtype>(0.0);

    size_t buffer_size = 0;
    void *buffer = nullptr;

    cusparse_zero_check(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y,
        value_type, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));

    if (buffer_size > 0) {
      zero_check(cudaMalloc(&buffer, buffer_size));
    }

    for (int warmup = 0; warmup < WARMUP; warmup++) {
      cusparse_zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat, x, &beta, y, value_type,
                                       CUSPARSE_SPMV_CSR_ALG1, buffer));

      zero_check(cudaGetLastError());
      zero_check(cudaDeviceSynchronize());
    }

    CUDA_EVENT_TIMER_DEF(KernelTime);

    for (int iter = 0; iter < NITER; iter++) {
      CUDA_START_EVENT_TIMER(KernelTime);

      cusparse_zero_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat, x, &beta, y, value_type,
                                       CUSPARSE_SPMV_CSR_ALG1, buffer));

      CUDA_STOP_EVENT_TIMER(KernelTime);

      float timer = 0.0f;
      CUDA_EVENT_TIMER_ELAPSED(KernelTime, timer);

      kernel_times[iter] = timer / 1e3;

      if (iter == 0) {
        first_run_time = kernel_times[iter];
      }

      zero_check(cudaGetLastError());
      zero_check(cudaDeviceSynchronize());
    }

    CUDA_EVENT_TIMER_DESTROY(KernelTime);

    zero_check(cudaMemcpy(local_result, d_local_result,
                          sizeof(vtype) * local_rows, cudaMemcpyDeviceToHost));

    if (buffer != nullptr) {
      zero_check(cudaFree(buffer));
    }

    cusparse_zero_check(cusparseDestroyDnVec(y));
    cusparse_zero_check(cusparseDestroyDnVec(x));
    cusparse_zero_check(cusparseDestroySpMat(mat));
    cusparse_zero_check(cusparseDestroy(handle));

    zero_check(cudaFree(d_csr_lrow));
    zero_check(cudaFree(d_aval));
    zero_check(cudaFree(d_local_result));
  }

  results.populate_kernel_times(kernel_times, NITER, first_run_time, rank,
                                comm_size);
  results.populate_flop_metrics(rank_nnz, rank, comm_size);

  free(local_dense_vec);

  zero_check(cudaFree(d_local_dense_vec));
  zero_check(cudaFree(d_global_acol));
  zero_check(cudaFree(d_editable_acol));

  if (d_extended_dense_vec != nullptr) {
    zero_check(cudaFree(d_extended_dense_vec));
  }

  int row_counts[comm_size];
  int row_displacements[comm_size];

  for (int i = 0; i < comm_size; i++) {
    row_counts[i] = compute_cyclic_count(header->rows, i, comm_size);
  }

  int total_rows =
      compute_displacements(row_counts, row_displacements, comm_size);
  assert(total_rows == header->rows);

  error =
      MPI_Gatherv(local_result, local_rows, MPI_FLOAT, rank_0_result_cyclic,
                  row_counts, row_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  free(local_result);

  if (rank == 0) {
    cyclic_to_global(rank_0_result, rank_0_result_cyclic, row_displacements,
                     row_counts, comm_size);

    double error = compute_mse(rank_0_result, local_cpu_result, header->rows);
    printf("error between computations (CSR, CUSPARSE): %10f\n", error);
    results.error = error;

    free(rank_0_result);
    free(rank_0_result_cyclic);
  }

  free(dense_vec);

  return results;
}
