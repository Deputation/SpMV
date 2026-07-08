#ifndef MPI_CPU_H
#define MPI_CPU_H

#include "constants.h"

void spmv_coo_local_out_extended(const idxtype *lrow, const idxtype *acol,
                                 const vtype *aval, idxtype nnz,
                                 const vtype *extended_dense_vec,
                                 vtype *local_result);

void spmv_csr_local_out_extended(const idxtype *csr_lrow, const idxtype *acol,
                                 const vtype *aval, const idxtype lrows,
                                 const vtype *extended_dense_vec, vtype *out);

#endif
