#ifndef CSR_H
#define CSR_H

#include <cstdint>

#include "constants.h"

__global__ void spmv_csr_naive(const idxtype *csr_arow, const idxtype *acol,
                               const vtype *aval, const idxtype rows,
                               const vtype *vec, vtype *out);

__global__ void spmv_csr_naive_fake(const idxtype *csr_arow,
                                    const idxtype *acol,
                                    const vtype *aval, const idxtype rows,
                                    const vtype *vec, vtype *out);

__global__ void spmv_csr_warp_row(const idxtype *csr_arow,
                                  const idxtype *acol, const vtype *aval,
                                  const idxtype rows, const vtype *vec,
                                  vtype *out);

__global__ void spmv_csr_warp_row_fake(const idxtype *csr_arow,
                                       const idxtype *acol,
                                       const vtype *aval,
                                       const idxtype rows,
                                       const vtype *vec, vtype *out);

#endif
