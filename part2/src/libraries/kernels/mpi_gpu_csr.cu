#include "gpu_utils.cuh"
#include "mpi_gpu_csr.cuh"

__global__ void spmv_csr_local_out_extended_kernel(
    const idxtype *csr_lrow, const idxtype *acol, const vtype *aval,
    const idxtype lrows, const vtype *extended_dense_vec, vtype *local_result) {
  idxtype idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // all threads inside a warp which will perform the warp-wise reduction will
  // need to be assigned to the same row, dividing the overall idx by the warp
  // size will let us assign a row to every warp
  idxtype assigned_row = idx / WARP_SIZE;
  // we will need to identify each thread in the warp
  idxtype tid_in_warp = threadIdx.x % WARP_SIZE;

  if (assigned_row >= lrows) {
    return;
  }

  // gather starting and end point for the row from the compressed arow array
  // this is only serviced once per warp, so we only count this rows times, not
  // rows * 32 times in the bw calculation
  idxtype csr_arow_start = csr_lrow[assigned_row];
  idxtype csr_arow_end = csr_lrow[assigned_row + 1];

  vtype result = 0;

  // every thread will compute the sum for the assigned row, with its tid in the
  // warp serving as an offset, striding the sum by warp size elements each time
  // to make sure threads don't collide with their sums
  for (idxtype j = csr_arow_start + tid_in_warp; j < csr_arow_end;
       j += WARP_SIZE) {
    result += aval[j] * extended_dense_vec[acol[j]];
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
    local_result[assigned_row] += result;
  }
}
