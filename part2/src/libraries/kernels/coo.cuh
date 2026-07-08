#ifndef COO_H
#define COO_H

#include "constants.h"

__global__ void spmv_coo_naive(const idxtype *arow, const idxtype *acol,
                               const vtype *aval, const idxtype nnz,
                               const idxtype rows, const vtype *vec,
                               vtype *out);

__global__ void spmv_coo_naive_fake(const idxtype *arow,
                                    const idxtype *acol,
                                    const vtype *aval, const idxtype nnz,
                                    const idxtype rows, const vtype *vec,
                                    vtype *out);

#endif
