#include "gpu_mpi_utils.cuh"
#include "mpi_benchmark_results.cuh"
#include "benchmark_values.h"
#include "host_mpi_utils.h"

void benchmark_exchange_columns(
    const idxtype *global_acol, idxtype *editable_acol, idxtype local_nnz,
    idxtype total_dense_vec_size, const vtype *local_dense_vec,
    int local_dense_count, int rank, int comm_size,
    vtype **out_extended_dense_vec, int *out_extended_dense_vec_count,
    int *out_received_columns_count, mpi_compute_benchmark_results *results) {
  *out_extended_dense_vec = nullptr;
  *out_extended_dense_vec_count = 0;
  *out_received_columns_count = 0;

  double ghost_exchange_times[NITER];
  for (size_t i = 0; i < NITER; i++) {
    ghost_exchange_times[i] = 0.0;
  }

  double ghost_exchange_alltoallv_times[NITER];
  for (size_t i = 0; i < NITER; i++) {
    ghost_exchange_alltoallv_times[i] = 0.0;
  }

  double first_run_time = 0.0;
  double first_run_alltoallv_time = 0.0;

  uint64_t ghost_values_received = 0;
  uint64_t ghost_bytes_sent = 0;
  uint64_t ghost_bytes_received = 0;

  for (int warmup = 0; warmup < WARMUP; warmup++) {
    memcpy(editable_acol, global_acol, sizeof(idxtype) * local_nnz);

    vtype *warmup_extended_dense_vec = nullptr;
    int warmup_extended_dense_vec_count = 0;
    int warmup_received_columns_count = 0;

    ghost_exchange_metrics metrics;

    exchange_columns(editable_acol, local_nnz, total_dense_vec_size,
                     local_dense_vec, local_dense_count, rank, comm_size,
                     &warmup_extended_dense_vec,
                     &warmup_extended_dense_vec_count,
                     &warmup_received_columns_count, &metrics);

    free(warmup_extended_dense_vec);
  }

  for (int iter = 0; iter < NITER; iter++) {
    if (*out_extended_dense_vec != nullptr) {
      free(*out_extended_dense_vec);
      *out_extended_dense_vec = nullptr;
    }

    *out_extended_dense_vec_count = 0;
    *out_received_columns_count = 0;

    memcpy(editable_acol, global_acol, sizeof(idxtype) * local_nnz);

    ghost_exchange_metrics metrics;

    exchange_columns(editable_acol, local_nnz, total_dense_vec_size,
                     local_dense_vec, local_dense_count, rank, comm_size,
                     out_extended_dense_vec, out_extended_dense_vec_count,
                     out_received_columns_count, &metrics);

    ghost_exchange_times[iter] = metrics.ghost_exchange_time;
    ghost_exchange_alltoallv_times[iter] =
        metrics.ghost_exchange_alltoallv_time;

    if (iter == 0) {
      first_run_time = metrics.ghost_exchange_time;
      first_run_alltoallv_time = metrics.ghost_exchange_alltoallv_time;
      ghost_values_received = metrics.ghost_values_received;
      ghost_bytes_sent = metrics.ghost_bytes_sent;
      ghost_bytes_received = metrics.ghost_bytes_received;
    } else {
      assert(ghost_values_received == metrics.ghost_values_received);
      assert(ghost_bytes_sent == metrics.ghost_bytes_sent);
      assert(ghost_bytes_received == metrics.ghost_bytes_received);
    }
  }

  results->populate_ghost_exchange_metrics(
      ghost_exchange_times, ghost_exchange_alltoallv_times, NITER,
      first_run_time, first_run_alltoallv_time, ghost_values_received,
      ghost_bytes_sent, ghost_bytes_received, rank, comm_size);
}

void benchmark_exchange_columns_gpu(
    const idxtype *d_global_acol,
    idxtype *d_editable_acol,
    idxtype local_nnz,
    idxtype total_dense_vec_size,
    const vtype *d_local_dense_vec,
    int local_dense_count,
    int rank,
    int comm_size,
    vtype **out_d_extended_dense_vec,
    int *out_extended_dense_vec_count,
    int *out_received_columns_count,
    mpi_compute_benchmark_results *results) {
  *out_d_extended_dense_vec = nullptr;
  *out_extended_dense_vec_count = 0;
  *out_received_columns_count = 0;

  double ghost_exchange_times[NITER];
  double ghost_exchange_alltoallv_times[NITER];

  for (int i = 0; i < NITER; i++) {
    ghost_exchange_times[i] = 0.0;
    ghost_exchange_alltoallv_times[i] = 0.0;
  }

  double first_run_time = 0.0;
  double first_run_alltoallv_time = 0.0;

  uint64_t ghost_values_received = 0;
  uint64_t ghost_bytes_sent = 0;
  uint64_t ghost_bytes_received = 0;

  for (int warmup = 0; warmup < WARMUP; warmup++) {
    if (local_nnz > 0) {
      zero_check(cudaMemcpy(d_editable_acol,
                            d_global_acol,
                            sizeof(idxtype) * local_nnz,
                            cudaMemcpyDeviceToDevice));
    }

    vtype *warmup_d_extended_dense_vec = nullptr;
    int warmup_extended_dense_vec_count = 0;
    int warmup_received_columns_count = 0;

    ghost_exchange_metrics metrics;

    exchange_columns_gpu(d_editable_acol,
                         local_nnz,
                         total_dense_vec_size,
                         d_local_dense_vec,
                         local_dense_count,
                         rank,
                         comm_size,
                         &warmup_d_extended_dense_vec,
                         &warmup_extended_dense_vec_count,
                         &warmup_received_columns_count,
                         &metrics);

    if (warmup_d_extended_dense_vec != nullptr) {
      zero_check(cudaFree(warmup_d_extended_dense_vec));
    }
  }

  for (int iter = 0; iter < NITER; iter++) {
    if (*out_d_extended_dense_vec != nullptr) {
      zero_check(cudaFree(*out_d_extended_dense_vec));
      *out_d_extended_dense_vec = nullptr;
    }

    *out_extended_dense_vec_count = 0;
    *out_received_columns_count = 0;

    if (local_nnz > 0) {
      zero_check(cudaMemcpy(d_editable_acol,
                            d_global_acol,
                            sizeof(idxtype) * local_nnz,
                            cudaMemcpyDeviceToDevice));
    }

    ghost_exchange_metrics metrics;

    exchange_columns_gpu(d_editable_acol,
                         local_nnz,
                         total_dense_vec_size,
                         d_local_dense_vec,
                         local_dense_count,
                         rank,
                         comm_size,
                         out_d_extended_dense_vec,
                         out_extended_dense_vec_count,
                         out_received_columns_count,
                         &metrics);

    ghost_exchange_times[iter] = metrics.ghost_exchange_time;
    ghost_exchange_alltoallv_times[iter] =
        metrics.ghost_exchange_alltoallv_time;

    if (iter == 0) {
      first_run_time = metrics.ghost_exchange_time;
      first_run_alltoallv_time = metrics.ghost_exchange_alltoallv_time;
      ghost_values_received = metrics.ghost_values_received;
      ghost_bytes_sent = metrics.ghost_bytes_sent;
      ghost_bytes_received = metrics.ghost_bytes_received;
    } else {
      assert(ghost_values_received == metrics.ghost_values_received);
      assert(ghost_bytes_sent == metrics.ghost_bytes_sent);
      assert(ghost_bytes_received == metrics.ghost_bytes_received);
    }
  }

  results->populate_ghost_exchange_metrics(
      ghost_exchange_times,
      ghost_exchange_alltoallv_times,
      NITER,
      first_run_time,
      first_run_alltoallv_time,
      ghost_values_received,
      ghost_bytes_sent,
      ghost_bytes_received,
      rank,
      comm_size);
}
