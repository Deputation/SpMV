#ifndef GPU_IMPLEMENTATIONS_H
#define GPU_IMPLEMENTATIONS_H

#include <cstdint>

#include "constants.h"

__global__ void spmv_coo_naive(const int64_t *arow, const int64_t *acol,
                               const valuetype *aval, const int64_t nnz,
                               const int64_t rows, const valuetype *vec,
                               valuetype *out);

__global__ void spmv_coo_naive_fake(const int64_t *arow, const int64_t *acol,
                                    const valuetype *aval, const int64_t nnz,
                                    const int64_t rows, const valuetype *vec,
                                    valuetype *out);

#endif
