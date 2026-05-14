#ifndef BENCHMARK_RESULTS_H
#define BENCHMARK_RESULTS_H

#include <cstddef>
#include <string.h>

// the following formulas were written by reading the implemented kernels line
// by line and carefully counting the bytes written and read whilst considering
// the amount of threads launched.
//
// we will use the geometric mean in these calculations as we want a bandwidth
// value that is resistant to outliers in the collected timing data
#define coo_formula                                                            \
  ((((2 * sizeof(int64_t)) + (4 * sizeof(valuetype))) * nnz) / 1e9) /          \
      fake_kernel_geo_mean_time
#define csr_formula                                                            \
  (((2 * sizeof(int64_t) * rows) + (1 * sizeof(int64_t) * nnz) +               \
    (4 * sizeof(valuetype) * nnz)) /                                           \
   1e9) /                                                                      \
      fake_kernel_geo_mean_time

// clang-format off
//
// operations multiplied only by rows are performed by a single thread (or by a warp,
// though the warp is serviced all at once, so we only count accesses to the compressed row pointers
// array as one request per row, despite a full warp of 32 threads making said access)
//
// operations multiplied by nnz are performed for all nnz values in the matrix
//
// in this formula, in order:
// reads from compressed row index arrays for all threads, no mul by 32, see above ((2 * sizeof(int64_t) * rows))
// read and write (2 accesses) to out array for thread at lane 0 ((2 * sizeof(valuetype) * rows))
// read from acol for all nnz (1 * sizeof(int64_t) * nnz)
// read from dense vec and aval for all nnz (2 * sizeof(valuetype) * nnz)
#define csr_warp_row_formula                                                   \
  (((2 * sizeof(int64_t) * rows) +                                      \
    (2 * sizeof(valuetype) * rows) +                                           \
    (1 * sizeof(int64_t) * nnz) +                                              \
    (2 * sizeof(valuetype) * nnz)) /                                           \
   1e9) /                                                                      \
      fake_kernel_geo_mean_time
// clang-format on

struct benchmark_results {
  char method[128];
  double time_average;
  double time_average_geo;
  double time_deviation;
  double error;
  double gflops;
  double effective_bw;
  double preprocessing_time_average;
  double preprocessing_time_average_geo;
  double preprocessing_time_deviation;

  benchmark_results() {
    strcpy(method, "N/A");
    time_average = -1.f;
    time_average_geo = -1.f;
    time_deviation = -1.f;
    error = -1.f;
    gflops = -1.f;
    effective_bw = -1.f;
    preprocessing_time_average = -1.f;
    preprocessing_time_average_geo = -1.f;
    preprocessing_time_deviation = -1.f;
  }
};

int write_results(const char *name, const char *mat_name,
                  const benchmark_results results[], const size_t elements);

#endif
