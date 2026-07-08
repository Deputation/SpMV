#include <omp.h>
#include <stdlib.h>

#include "constants.h"
#include "cpu.h"

void spmv_coo(const idxtype *arow, const idxtype *acol,
              const vtype *aval, const idxtype nnz, const vtype *vec,
              vtype *out) {
  // OpenMP is unlikely to speed things up here.
#pragma omp parallel for
  for (idxtype i = 0; i < nnz; i++) {
// read from arow[i] * sizeof(indextype)
// read from acol[i] * sizeof(indextype)
// read from aval[i] * sizeof(valuetype)
// read from vec[acol[i]] * sizeof(valuetype)
// read from out[arow[i]] * sizeof(valuetype)
// write to out[arow[i]] * sizeof(valuetype)
// (3 * sizeof(indextype)) + (2 * sizeof(valuetype))
#pragma omp atomic
    out[arow[i]] += aval[i] * vec[acol[i]];
  }
}

void spmv_csr(const idxtype *csr_arow, const idxtype *acol,
              const vtype *aval, const idxtype rows, const vtype *vec,
              vtype *out) {
#pragma omp parallel for
  for (idxtype i = 0; i < rows; i++) {
    // read from csr_arow[i] * sizeof(indextype)
    // read from csr_arow[i + 1] * sizeof(indextype)
    // read from out[i] * sizeof(valuetype)
    vtype sum = 0;
    for (idxtype j = csr_arow[i]; j < csr_arow[i + 1]; j++) {
      // read from aval[j] * sizeof(indextype)
      // read from acol[j] * sizeof(indextype)
      // read from vec[acol[j]] * sizeof(valuetype)

      // write to out[i] * sizeof (valuetype)
      sum += aval[j] * vec[acol[j]];
    }
    out[i] = sum;
  }
  // (2 reads * rows) + ((3 reads + 1 write) * nnz)
  //
  // (2 * sizeof(indextype) * rows) + (1 * sizeof(valuetype) * rows) + (2 *
  // sizeof(indextype) * nnz) + (2 * sizeof(valuetype) * nnz) accesses
}
