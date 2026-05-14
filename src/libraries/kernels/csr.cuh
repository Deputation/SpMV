#ifndef CSR_H
#define CSR_H

#include <cstdint>

#include "constants.h"

__global__ void spmv_csr_naive(const int64_t *csr_arow, const int64_t *acol,
                               const valuetype *aval, const int64_t rows,
                               const valuetype *vec, valuetype *out);

__global__ void spmv_csr_naive_fake(const int64_t *csr_arow,
                                    const int64_t *acol, const valuetype *aval,
                                    const int64_t rows, const valuetype *vec,
                                    valuetype *out);

__global__ void spmv_csr_warp_row(const int64_t *csr_arow, const int64_t *acol,
                                  const valuetype *aval, const int64_t rows,
                                  const valuetype *vec, valuetype *out);

__global__ void spmv_csr_warp_row_fake(const int64_t *csr_arow,
                                       const int64_t *acol,
                                       const valuetype *aval,
                                       const int64_t rows, const valuetype *vec,
                                       valuetype *out);

#endif
