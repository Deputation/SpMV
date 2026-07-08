#ifndef MPI_GPU_COO_H
#define MPI_GPU_COO_H

#include "constants.h"

__global__ void spmv_coo_local_out_extended_kernel(
    const idxtype *lrow, const idxtype *acol, const vtype *aval, idxtype nnz,
    const vtype *extended_dense_vec, vtype *local_result);

#endif
