#ifndef MPI_GPU_CSR_H
#define MPI_GPU_CSR_H

#include "constants.h"

__global__ void spmv_csr_local_out_extended_kernel(
    const idxtype *csr_lrow, const idxtype *acol, const vtype *aval,
    const idxtype lrows, const vtype *extended_dense_vec, vtype *local_result);

#endif
