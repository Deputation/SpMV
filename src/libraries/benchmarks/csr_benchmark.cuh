#ifndef CSR_BENCHMARK_H
#define CSR_BENCHMARK_H

#include "benchmark_results.h"
#include "constants.h"

#include <cstdint>
#include <cusparse.h>

benchmark_results
benchmark_csr_cpu(const int64_t *host_csr_arow, const int64_t *host_acol,
                  const valuetype *host_aval, const int64_t rows,
                  const int64_t cols, const int64_t nnz,
                  valuetype **ptr_to_dev_out_csr, const valuetype *dev_dense_vec);

benchmark_results
benchmark_csr_scalar(const int64_t *csr_arow, const int64_t *acol,
                           const valuetype *aval, const int64_t rows,
                           const int64_t nnz, const valuetype *out_csr_cpu,
                           const valuetype *dense_vec);

benchmark_results
benchmark_csr_vector_warp(const int64_t *csr_arow, const int64_t *acol,
                              const valuetype *aval, const int64_t rows,
                              const int64_t nnz, const valuetype *out_csr_cpu,
                              const valuetype *dense_vec);

benchmark_results benchmark_cusparse_csr_gpu(
    const cusparseHandle_t handle, const int64_t *csr_arow, const int64_t *acol,
    const valuetype *aval, const int64_t rows, const int64_t cols,
    const int64_t nnz, const valuetype *out_csr_cpu,
    const valuetype *dense_vec);
#endif
