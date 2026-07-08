#ifndef CPU_H
#define CPU_H

#include <cstdint>

#include "constants.h"

void spmv_coo(const idxtype *arow, const idxtype *acol,
              const vtype *aval, const idxtype nnz, const vtype *vec,
              vtype *out);

void spmv_csr(const idxtype *csr_arow, const idxtype *acol,
              const vtype *aval, const idxtype rows, const vtype *vec,
              vtype *out);

#endif
