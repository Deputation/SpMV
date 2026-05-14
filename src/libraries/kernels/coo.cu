#include "coo.cuh"

// naive implementation, every thread takes its own nnz in the coo format and
// performs part of the computation, results are stored in the global memory
// with a locking mechanism, effectively serialising the operation in the worst
// case.
__global__ void spmv_coo_naive(const int64_t *arow, const int64_t *acol,
                               const valuetype *aval, const int64_t nnz,
                               const int64_t rows, const valuetype *vec,
                               valuetype *out) {
  int64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= nnz) {
    return;
  }

  atomicAdd(&out[arow[idx]], aval[idx] * vec[acol[idx]]);
}

// fake kernel used to compute effective bandwidth
__global__ void spmv_coo_naive_fake(const int64_t *arow, const int64_t *acol,
                                    const valuetype *aval, const int64_t nnz,
                                    const int64_t rows, const valuetype *vec,
                                    valuetype *out) {
  int64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= nnz) {
    return;
  }

  // variables marked as volatile to stop the optimiser
  volatile int64_t acol_read = acol[idx];
  volatile valuetype aval_read = aval[idx];
  volatile valuetype vec_read = vec[acol_read];

  // assuming the optimiser does not think it knows better, this does not need
  // to be volatile
  int64_t arow_read = arow[idx];
  // simulate atomic access
  atomicAdd(&out[arow_read], 1);
}
