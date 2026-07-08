#include "mpi_gpu_coo.cuh"

__global__ void spmv_coo_local_out_extended_kernel(
    const idxtype *lrow,
    const idxtype *acol,
    const vtype *aval,
    idxtype nnz,
    const vtype *extended_dense_vec,
    vtype *local_result) {
  idxtype i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nnz) {
    return;
  }

  atomicAdd(&local_result[lrow[i]],
            aval[i] * extended_dense_vec[acol[i]]);
}
