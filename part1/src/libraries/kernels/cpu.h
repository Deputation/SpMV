#ifndef CPU_IMPLEMENTATIONS_H
#define CPU_IMPLEMENTATIONS_H

#include <cstdint>

#include "constants.h"

void spmv_coo(const int64_t *arow, const int64_t *acol, const valuetype *aval,
              const int64_t nnz, const valuetype *vec, valuetype *out);
void spmv_csr(const int64_t *csr_arow, const int64_t *acol,
              const valuetype *aval, const int64_t rows, const valuetype *vec,
              valuetype *out);

#endif
