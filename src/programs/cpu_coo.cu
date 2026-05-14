#include "cpu.h"
#include "host_utils.cuh"
#include "parse_file.cuh"

#include <stdio.h>
#include <assert.h>

void test_coo_cpu_host(const int64_t *arow, const int64_t *acol,
                       const valuetype *aval, const int64_t rows,
                       const int64_t nnz, const valuetype *dense_vec) {
  valuetype *out_coo = (valuetype *)malloc(sizeof(valuetype) * rows);
  assert(out_coo != nullptr);
  memset(out_coo, 0, sizeof(valuetype) * rows);
  spmv_coo(arow, acol, aval, nnz, dense_vec, out_coo);
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

  int64_t *host_acol, *host_arow;
  valuetype *host_aval;
  int64_t rows, cols, nnz;
  if (parse_file_coo(argv[1], &host_arow, &host_acol, &host_aval, &rows, &cols,
                     &nnz) != 0) {
    printf("error encountered when parsing file\n");
    return 2;
  }

  TIMER_STOP(ParseTime);

  double parse_time = TIMER_ELAPSED(ParseTime) / 1e6;

  printf("parsed in %10fs\n", parse_time);

  valuetype *host_dense_vec = init_dense_vec_host(cols);
  bench_expression(test_coo_cpu_host(host_arow, host_acol, host_aval, rows, nnz,
                                     host_dense_vec));

  free(host_dense_vec);

  free(host_acol);
  free(host_arow);
  free(host_aval);

  printf("done\n");

  return 0;
}
