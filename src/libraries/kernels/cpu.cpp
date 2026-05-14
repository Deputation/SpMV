#include <omp.h>
#include <stdlib.h>

#include "cpu.h"
#include "constants.h"

void spmv_coo(const int64_t *arow, const int64_t *acol, const valuetype *aval,
              const int64_t nnz, const valuetype *vec, valuetype *out) {
  // OpenMP is unlikely to speed things up here.
#pragma omp parallel for
  for (int64_t i = 0; i < nnz; i++) {
// read from arow[i] * sizeof(int64_t)
// read from acol[i] * sizeof(int64_t)
// read from aval[i] * sizeof(valuetype)
// read from vec[acol[i]] * sizeof(valuetype)
// read from out[arow[i]] * sizeof(valuetype)
// write to out[arow[i]] * sizeof(valuetype)
// (3 * sizeof(int64_t)) + (2 * sizeof(valuetype))
#pragma omp atomic
    out[arow[i]] += aval[i] * vec[acol[i]];
  }
}

void spmv_csr(const int64_t *csr_arow, const int64_t *acol,
              const valuetype *aval, const int64_t rows, const valuetype *vec,
              valuetype *out) {
#pragma omp parallel for
  for (int64_t i = 0; i < rows; i++) {
    // read from csr_arow[i] * sizeof(int64_t)
    // read from csr_arow[i + 1] * sizeof(int64_t)
    // read from out[i] * sizeof(valuetype)
    valuetype sum = 0;
    for (int64_t j = csr_arow[i]; j < csr_arow[i + 1]; j++) {
      // read from aval[j] * sizeof(int64_t)
      // read from acol[j] * sizeof(int64_t)
      // read from vec[acol[j]] * sizeof(valuetype)

      // write to out[i] * sizeof (valuetype)
      sum += aval[j] * vec[acol[j]];
    }
    out[i] = sum;
  }
  // (2 reads * rows) + ((3 reads + 1 write) * nnz)
  //
  // (2 * sizeof(int64_t) * rows) + (1 * sizeof(valuetype) * rows) + (2 *
  // sizeof(int64_t) * nnz) + (2 * sizeof(valuetype) * nnz) accesses
}
