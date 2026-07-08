#include "gpu_mpi_utils.cuh"
#include "host_utils.cuh"
#include "parallel_parse_file.h"

#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#include <limits>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

template <typename T> T *cuda_alloc_at_least_one(size_t count) {
  T *ptr = nullptr;
  size_t alloc_count = count > 0 ? count : 1;
  zero_check(
      cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(T) * alloc_count));
  return ptr;
}

static MPI_Datatype mpi_idxtype() {
  static_assert(std::is_signed<idxtype>::value,
                "idxtype is expected to be signed");

  if (sizeof(idxtype) == sizeof(int64_t)) {
    return MPI_INT64_T;
  }

  if (sizeof(idxtype) == sizeof(int32_t)) {
    return MPI_INT32_T;
  }

  fprintf(stderr, "unsupported idxtype size for MPI\n");
  MPI_Abort(MPI_COMM_WORLD, 1);
  return MPI_DATATYPE_NULL;
}

static MPI_Datatype mpi_vtype() {
  if (sizeof(vtype) == sizeof(float)) {
    return MPI_FLOAT;
  }

  if (sizeof(vtype) == sizeof(double)) {
    return MPI_DOUBLE;
  }

  fprintf(stderr, "unsupported vtype size for MPI\n");
  MPI_Abort(MPI_COMM_WORLD, 1);
  return MPI_DATATYPE_NULL;
}

__global__ void make_remote_keys_kernel(const idxtype *d_acol, idxtype *d_keys,
                                        idxtype local_nnz,
                                        idxtype total_dense_vec_size, int rank,
                                        int comm_size, idxtype sentinel) {
  idxtype i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= local_nnz) {
    return;
  }

  idxtype col = d_acol[i];

  assert(col >= 0);
  assert(col < total_dense_vec_size);

  int owner = static_cast<int>(col % comm_size);

  if (owner == rank) {
    d_keys[i] = sentinel;
  } else {
    d_keys[i] = static_cast<idxtype>(owner) * total_dense_vec_size + col;
  }
}

struct key_to_global_col {
  idxtype total_dense_vec_size;

  __host__ __device__ idxtype operator()(idxtype key) const {
    return key % total_dense_vec_size;
  }
};

__global__ void gather_requested_values_kernel(
    const idxtype *d_columns_to_send, int count, int comm_size, int rank,
    idxtype total_dense_vec_size, const vtype *d_local_dense_vec,
    int local_dense_count, vtype *d_values_to_send) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= count) {
    return;
  }

  idxtype global_col = d_columns_to_send[i];

  assert(global_col >= 0);
  assert(global_col < total_dense_vec_size);
  assert(static_cast<int>(global_col % comm_size) == rank);

  idxtype local_col = global_col / comm_size;

  assert(local_col >= 0);
  assert(local_col < static_cast<idxtype>(local_dense_count));

  d_values_to_send[i] = d_local_dense_vec[local_col];
}

__global__ void remap_acol_kernel(idxtype *d_acol, idxtype local_nnz,
                                  const idxtype *d_remote_keys,
                                  const int *d_recv_counts,
                                  const int *d_recv_displs,
                                  idxtype total_dense_vec_size,
                                  int local_dense_count, int rank,
                                  int comm_size, int extended_dense_vec_count) {
  idxtype i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= local_nnz) {
    return;
  }

  idxtype global_col = d_acol[i];

  assert(global_col >= 0);
  assert(global_col < total_dense_vec_size);

  int owner = static_cast<int>(global_col % comm_size);

  if (owner == rank) {
    idxtype local_col = global_col / comm_size;

    assert(local_col >= 0);
    assert(local_col < static_cast<idxtype>(local_dense_count));

    d_acol[i] = local_col;
  } else {
    idxtype key =
        static_cast<idxtype>(owner) * total_dense_vec_size + global_col;

    int base = d_recv_displs[owner];
    int count = d_recv_counts[owner];

    int lo = 0;
    int hi = count;

    while (lo < hi) {
      int mid = lo + ((hi - lo) / 2);
      idxtype mid_key = d_remote_keys[base + mid];

      if (mid_key < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    assert(lo < count);
    assert(d_remote_keys[base + lo] == key);

    int ghost_slot = base + lo;
    d_acol[i] = static_cast<idxtype>(local_dense_count + ghost_slot);
  }

  assert(d_acol[i] >= 0);
  assert(d_acol[i] < static_cast<idxtype>(extended_dense_vec_count));
}

void exchange_columns_gpu(idxtype *d_editable_acol, idxtype local_nnz,
                          idxtype total_dense_vec_size,
                          const vtype *d_local_dense_vec, int local_dense_count,
                          int rank, int comm_size,
                          vtype **out_d_extended_dense_vec,
                          int *out_extended_dense_vec_count,
                          int *out_received_columns_count,
                          ghost_exchange_metrics *metrics) {
  if (metrics != nullptr) {
    metrics->ghost_exchange_time = 0.0;
    metrics->ghost_exchange_alltoallv_time = 0.0;
    metrics->ghost_values_received = 0;
    metrics->ghost_bytes_sent = 0;
    metrics->ghost_bytes_received = 0;
  }

  *out_d_extended_dense_vec = nullptr;
  *out_extended_dense_vec_count = 0;

  if (out_received_columns_count != nullptr) {
    *out_received_columns_count = 0;
  }

  const idxtype original_local_nnz = local_nnz;

  double ghost_exchange_t0 = MPI_Wtime();

  int error = MPI_SUCCESS;
  MPI_Datatype MPI_IDXTYPE = mpi_idxtype();
  MPI_Datatype MPI_VTYPE = mpi_vtype();

  const idxtype sentinel = std::numeric_limits<idxtype>::max();

  idxtype *d_remote_keys =
      cuda_alloc_at_least_one<idxtype>(static_cast<size_t>(original_local_nnz));

  int cols_to_recv_total = 0;

  if (original_local_nnz > 0) {
    int blocks = static_cast<int>((original_local_nnz + THREADS_PER_BLOCK - 1) /
                                  THREADS_PER_BLOCK);

    make_remote_keys_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_editable_acol, d_remote_keys, original_local_nnz,
        total_dense_vec_size, rank, comm_size, sentinel);

    zero_check(cudaGetLastError());

    thrust::sort(thrust::device, d_remote_keys,
                 d_remote_keys + original_local_nnz);

    auto valid_end =
        thrust::lower_bound(thrust::device, d_remote_keys,
                            d_remote_keys + original_local_nnz, sentinel);

    auto unique_end = thrust::unique(thrust::device, d_remote_keys, valid_end);

    cols_to_recv_total = static_cast<int>(unique_end - d_remote_keys);
  }

  std::vector<int> cols_to_recv_count(comm_size, 0);
  std::vector<int> cols_to_recv_displs(comm_size, 0);

  for (int owner = 0; owner < comm_size; owner++) {
    idxtype lower = static_cast<idxtype>(owner) * total_dense_vec_size;
    idxtype upper = static_cast<idxtype>(owner + 1) * total_dense_vec_size;

    auto begin = thrust::lower_bound(thrust::device, d_remote_keys,
                                     d_remote_keys + cols_to_recv_total, lower);

    auto end = thrust::lower_bound(thrust::device, d_remote_keys,
                                   d_remote_keys + cols_to_recv_total, upper);

    cols_to_recv_count[owner] = static_cast<int>(end - begin);
  }

  int checked_recv_total = compute_displacements(
      cols_to_recv_count.data(), cols_to_recv_displs.data(), comm_size);

  assert(checked_recv_total == cols_to_recv_total);

  idxtype *d_columns_needed =
      cuda_alloc_at_least_one<idxtype>(cols_to_recv_total);

  if (cols_to_recv_total > 0) {
    thrust::transform(thrust::device, d_remote_keys,
                      d_remote_keys + cols_to_recv_total, d_columns_needed,
                      key_to_global_col{total_dense_vec_size});
  }

  std::vector<int> cols_to_send_count(comm_size, 0);

  error = MPI_Alltoall(cols_to_recv_count.data(), 1, MPI_INT,
                       cols_to_send_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  std::vector<int> cols_to_send_displs(comm_size, 0);

  int cols_to_send_total = compute_displacements(
      cols_to_send_count.data(), cols_to_send_displs.data(), comm_size);

  uint64_t ghost_values_received = static_cast<uint64_t>(cols_to_recv_total);

  uint64_t ghost_bytes_sent = static_cast<uint64_t>(cols_to_recv_total) *
                                  static_cast<uint64_t>(sizeof(idxtype)) +
                              static_cast<uint64_t>(cols_to_send_total) *
                                  static_cast<uint64_t>(sizeof(vtype));

  uint64_t ghost_bytes_received = static_cast<uint64_t>(cols_to_send_total) *
                                      static_cast<uint64_t>(sizeof(idxtype)) +
                                  static_cast<uint64_t>(cols_to_recv_total) *
                                      static_cast<uint64_t>(sizeof(vtype));

  idxtype *d_columns_to_send =
      cuda_alloc_at_least_one<idxtype>(cols_to_send_total);

  zero_check(cudaDeviceSynchronize());

  double column_ids_alltoallv_t0 = MPI_Wtime();

  error = MPI_Alltoallv(
      d_columns_needed, cols_to_recv_count.data(), cols_to_recv_displs.data(),
      MPI_IDXTYPE, d_columns_to_send, cols_to_send_count.data(),
      cols_to_send_displs.data(), MPI_IDXTYPE, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  double column_ids_alltoallv_t1 = MPI_Wtime();

  vtype *d_values_to_send = cuda_alloc_at_least_one<vtype>(cols_to_send_total);

  if (cols_to_send_total > 0) {
    int blocks =
        (cols_to_send_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    gather_requested_values_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_columns_to_send, cols_to_send_total, comm_size, rank,
        total_dense_vec_size, d_local_dense_vec, local_dense_count,
        d_values_to_send);

    zero_check(cudaGetLastError());
  }

  vtype *d_values_received = cuda_alloc_at_least_one<vtype>(cols_to_recv_total);

  zero_check(cudaDeviceSynchronize());

  double values_alltoallv_t0 = MPI_Wtime();

  error = MPI_Alltoallv(d_values_to_send, cols_to_send_count.data(),
                        cols_to_send_displs.data(), MPI_VTYPE,
                        d_values_received, cols_to_recv_count.data(),
                        cols_to_recv_displs.data(), MPI_VTYPE, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  double values_alltoallv_t1 = MPI_Wtime();

  int extended_dense_vec_count = local_dense_count + cols_to_recv_total;

  vtype *d_extended_dense_vec =
      cuda_alloc_at_least_one<vtype>(extended_dense_vec_count);

  if (local_dense_count > 0) {
    zero_check(cudaMemcpy(d_extended_dense_vec, d_local_dense_vec,
                          sizeof(vtype) * local_dense_count,
                          cudaMemcpyDeviceToDevice));
  }

  if (cols_to_recv_total > 0) {
    zero_check(cudaMemcpy(d_extended_dense_vec + local_dense_count,
                          d_values_received, sizeof(vtype) * cols_to_recv_total,
                          cudaMemcpyDeviceToDevice));
  }

  int *d_recv_counts = cuda_alloc_at_least_one<int>(comm_size);
  int *d_recv_displs = cuda_alloc_at_least_one<int>(comm_size);

  zero_check(cudaMemcpy(d_recv_counts, cols_to_recv_count.data(),
                        sizeof(int) * comm_size, cudaMemcpyHostToDevice));

  zero_check(cudaMemcpy(d_recv_displs, cols_to_recv_displs.data(),
                        sizeof(int) * comm_size, cudaMemcpyHostToDevice));

  if (original_local_nnz > 0) {
    int blocks = static_cast<int>((original_local_nnz + THREADS_PER_BLOCK - 1) /
                                  THREADS_PER_BLOCK);

    remap_acol_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_editable_acol, original_local_nnz, d_remote_keys, d_recv_counts,
        d_recv_displs, total_dense_vec_size, local_dense_count, rank, comm_size,
        extended_dense_vec_count);

    zero_check(cudaGetLastError());
  }

  zero_check(cudaDeviceSynchronize());

  zero_check(cudaFree(d_remote_keys));
  zero_check(cudaFree(d_columns_needed));
  zero_check(cudaFree(d_columns_to_send));
  zero_check(cudaFree(d_values_to_send));
  zero_check(cudaFree(d_values_received));
  zero_check(cudaFree(d_recv_counts));
  zero_check(cudaFree(d_recv_displs));

  *out_d_extended_dense_vec = d_extended_dense_vec;
  *out_extended_dense_vec_count = extended_dense_vec_count;

  if (out_received_columns_count != nullptr) {
    *out_received_columns_count = cols_to_recv_total;
  }

  double ghost_exchange_t1 = MPI_Wtime();

  if (metrics != nullptr) {
    metrics->ghost_exchange_time = ghost_exchange_t1 - ghost_exchange_t0;
    metrics->ghost_exchange_alltoallv_time =
        (column_ids_alltoallv_t1 - column_ids_alltoallv_t0) +
        (values_alltoallv_t1 - values_alltoallv_t0);
    metrics->ghost_values_received = ghost_values_received;
    metrics->ghost_bytes_sent = ghost_bytes_sent;
    metrics->ghost_bytes_received = ghost_bytes_received;
  }
}
