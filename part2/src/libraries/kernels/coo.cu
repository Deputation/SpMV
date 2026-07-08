#include "coo.cuh"

// naive implementation, every thread takes its own nnz in the coo format and
// performs part of the computation, results are stored in the global memory
// with a locking mechanism, effectively serialising the operation in the worst
// case.
__global__ void spmv_coo_naive(const idxtype *arow, const idxtype *acol,
                               const vtype *aval, const idxtype nnz,
                               const idxtype rows, const vtype *vec,
                               vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= nnz) {
    return;
  }

  atomicAdd(&out[arow[idx]], aval[idx] * vec[acol[idx]]);
}

// fake kernel used to compute effective bandwidth
__global__ void spmv_coo_naive_fake(const idxtype *arow,
                                    const idxtype *acol,
                                    const vtype *aval, const idxtype nnz,
                                    const idxtype rows, const vtype *vec,
                                    vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= nnz) {
    return;
  }

  // variables marked as volatile to stop the optimiser
  volatile idxtype acol_read = acol[idx];
  volatile vtype aval_read = aval[idx];
  volatile vtype vec_read = vec[acol_read];

  // assuming the optimiser does not think it knows better, this does not need
  // to be volatile
  idxtype arow_read = arow[idx];
  // simulate atomic access
  atomicAdd(&out[arow_read], 1);
}
