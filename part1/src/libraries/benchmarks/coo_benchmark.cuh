#ifndef COO_BENCHMARK_H
#define COO_BENCHMARK_H

#include "benchmark_results.h"
#include "constants.h"

#include <cstdint>
#include <cusparse.h>

benchmark_results
benchmark_coo_cpu(const int64_t *host_arow, const int64_t *host_acol,
                  const valuetype *host_aval, const int64_t rows,
                  const int64_t cols, const int64_t nnz,
                  valuetype **ptr_to_dev_out_coo, const valuetype *dev_dense_vec);

benchmark_results
benchmark_coo_kernel_naive(const int64_t *dev_arow, const int64_t *dev_acol,
                           const valuetype *dev_aval, const int64_t rows,
                           const int64_t nnz, const valuetype *dev_out_coo,
                           const valuetype *dev_dense_vec);

benchmark_results benchmark_cusparse_coo_gpu(
    const cusparseHandle_t handle, const int64_t *dev_arow,
    const int64_t *dev_acol, const valuetype *dev_aval, const int64_t rows,
    const int64_t cols, const int64_t nnz, const valuetype *dev_out_coo,
    const valuetype *dev_dense_vec);

#endif
