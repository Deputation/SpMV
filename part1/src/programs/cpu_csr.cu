#include "cpu.h"
#include "host_utils.cuh"
#include "parse_file.cuh"

#include <stdio.h>
#include <assert.h>

void test_csr_cpu_host(const int64_t *csr_arow, const int64_t *acol,
                       const valuetype *aval, const int64_t rows,
                       const valuetype *dense_vec) {
  valuetype *out_csr = (valuetype *)malloc(sizeof(valuetype) * rows);
  assert(out_csr != nullptr);
  memset(out_csr, 0, sizeof(valuetype) * rows);
  spmv_csr(csr_arow, acol, aval, rows, dense_vec, out_csr);
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <matrix file>\n", argv[0]);
    return 1;
  }

  srand(42);

  printf("%s: working on %s\n", __FILE__, argv[1]);

  TIMER_DEF(ParseTime);

  TIMER_START(ParseTime);

  int64_t *host_acol, *host_csr_arow;
  valuetype *host_aval;
  int64_t rows, cols, nnz;
  if (parse_file_csr(argv[1], &host_csr_arow, &host_acol, &host_aval, &rows,
                     &cols, &nnz) != 0) {
    printf("error encountered when parsing file\n");
    return 2;
  }

  TIMER_STOP(ParseTime);

  double parse_time = TIMER_ELAPSED(ParseTime) / 1e6;

  printf("parsed in %10fs\n", parse_time);

  valuetype *host_dense_vec = init_dense_vec_host(cols);

  bench_expression(test_csr_cpu_host(host_csr_arow, host_acol, host_aval, rows,
                                     host_dense_vec));

  free(host_dense_vec);

  free(host_acol);
  free(host_csr_arow);
  free(host_aval);

  printf("done\n");

  return 0;
}
