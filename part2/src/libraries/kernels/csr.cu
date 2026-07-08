#include "csr.cuh"
#include "gpu_utils.cuh"

// fake kernel used to compute effective bandwidth
__global__ void spmv_csr_naive_fake(const idxtype *csr_arow,
                                    const idxtype *acol,
                                    const vtype *aval, const idxtype rows,
                                    const vtype *vec, vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx >= rows) {
    return;
  }

  for (idxtype j = csr_arow[idx]; j < csr_arow[idx + 1]; j++) {
    // everything has to be volatile here
    volatile idxtype acol_read = acol[j];
    volatile vtype aval_read = aval[j];
    volatile vtype vec_read = vec[acol_read];
    // simulate write with += as we want to fetch, increment then write
    out[idx] += 1;
  }
}

__global__ void spmv_csr_naive(const idxtype *csr_arow, const idxtype *acol,
                               const vtype *aval, const idxtype rows,
                               const vtype *vec, vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx >= rows) {
    return;
  }

  for (idxtype j = csr_arow[idx]; j < csr_arow[idx + 1]; j++) {
    out[idx] += aval[j] * vec[acol[j]];
  }
}

__global__ void spmv_csr_warp_row(const idxtype *csr_arow,
                                  const idxtype *acol, const vtype *aval,
                                  const idxtype rows, const vtype *vec,
                                  vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // all threads inside a warp which will perform the warp-wise reduction will
  // need to be assigned to the same row, dividing the overall idx by the warp
  // size will let us assign a row to every warp
  idxtype assigned_row = idx / WARP_SIZE;
  // we will need to identify each thread in the warp
  idxtype tid_in_warp = threadIdx.x % WARP_SIZE;

  if (assigned_row >= rows) {
    return;
  }

  // gather starting and end point for the row from the compressed arow array
  // this is only serviced once per warp, so we only count this rows times, not
  // rows * 32 times in the bw calculation
  idxtype csr_arow_start = csr_arow[assigned_row];
  idxtype csr_arow_end = csr_arow[assigned_row + 1];

  vtype result = 0;

  // every thread will compute the sum for the assigned row, with its tid in the
  // warp serving as an offset, striding the sum by warp size elements each time
  // to make sure threads don't collide with their sums
  for (idxtype j = csr_arow_start + tid_in_warp; j < csr_arow_end;
       j += WARP_SIZE) {
    result += aval[j] * vec[acol[j]];
  }

  // perform a reduction over threads in the warp, this takes log of warp size
  // steps (5 steps for a warp size of 32)
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // we use a mask comprised of ones as we do not expect divergence at the
    // starting if statement, it will suppress whole warps
    result += __shfl_down_sync(0xffffffff, result, offset);
  }

  // this reduces in the opposite direction wrt the coo base opt kernel, so
  // lane 0 has the value, given this warp is the only warp with
  // this row assigned to it, and given lane 0 has the computed value, we can
  // simply write to the global memory without worrying about atomic access
  // (provided we check that we're doing this from lane 0)
  if (tid_in_warp == 0) {
    out[assigned_row] += result;
  }
}

__global__ void spmv_csr_warp_row_fake(const idxtype *csr_arow,
                                       const idxtype *acol,
                                       const vtype *aval,
                                       const idxtype rows,
                                       const vtype *vec, vtype *out) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  idxtype assigned_row = idx / WARP_SIZE;
  idxtype tid_in_warp = threadIdx.x % WARP_SIZE;

  if (assigned_row >= rows) {
    return;
  }

  idxtype csr_arow_start = csr_arow[assigned_row];
  idxtype csr_arow_end = csr_arow[assigned_row + 1];

  vtype result = 0;

  for (idxtype j = csr_arow_start + tid_in_warp; j < csr_arow_end;
       j += WARP_SIZE) {
    volatile idxtype acol_read = acol[j];
    volatile vtype aval_read = aval[j];
    volatile vtype vec_read = vec[acol_read];
    // i have to do something in this loop, otherwise the compiler keeps
    // removing it, giving me unreasonably high bandwidth results.
    result += 1;
  }

  // we are not simulating this because we're interested in its math, but
  // because we want to induce the same kind of behaviour the kernel has when
  // performing it; that is we want to reproduce the situation in which certain
  // threads reach this loop first and wait on others to finish their longer
  // rows before performing the intra warp thread reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    result += __shfl_down_sync(0xffffffff, result, offset);
  }

  if (tid_in_warp == 0) {
    // a read and a write of valuetype type, this happens for nnz times
    out[assigned_row] += result;
  }
}
