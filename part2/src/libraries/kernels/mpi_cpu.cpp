#include "mpi_cpu.h"

void spmv_coo_local_out_extended(const idxtype *lrow, const idxtype *acol,
                                 const vtype *aval, idxtype nnz,
                                 const vtype *extended_dense_vec,
                                 vtype *local_result) {
#pragma omp parallel for
  for (idxtype i = 0; i < nnz; i++) {
#pragma omp atomic
    local_result[lrow[i]] += aval[i] * extended_dense_vec[acol[i]];
  }
}

void spmv_csr_local_out_extended(const idxtype *csr_lrow, const idxtype *acol,
                                 const vtype *aval, const idxtype lrows,
                                 const vtype *extended_dense_vec, vtype *out) {
#pragma omp parallel for
  for (idxtype i = 0; i < lrows; i++) {
    vtype sum = 0;
    for (idxtype j = csr_lrow[i]; j < csr_lrow[i + 1]; j++) {
      sum += aval[j] * extended_dense_vec[acol[j]];
    }
    out[i] = sum;
  }
}
