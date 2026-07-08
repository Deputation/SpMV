#include "csr_benchmark.cuh"
#include "parse_file.cuh"
#include "host_utils.cuh"
#include "gpu_utils.cuh"

#include <cusparse.h>
#include <stdio.h>

void allocate_device_buffers(int64_t *host_csr_arow, int64_t *host_acol,
                             valuetype *host_aval, int64_t rows,
                             int64_t nnz, int64_t **dev_acol,
                             int64_t **dev_csr_arow, valuetype **dev_aval) {
  zero_check(cudaMalloc(dev_csr_arow, sizeof(int64_t) * (rows + 1)));
  zero_check(cudaMalloc(dev_acol, sizeof(int64_t) * nnz));
  zero_check(cudaMalloc(dev_aval, sizeof(valuetype) * nnz));

  zero_check(cudaMemcpy(*dev_csr_arow, host_csr_arow,
                        sizeof(int64_t) * (rows + 1), cudaMemcpyHostToDevice));
  zero_check(cudaMemcpy(*dev_acol, host_acol, sizeof(int64_t) * nnz,
                        cudaMemcpyHostToDevice));
  zero_check(cudaMemcpy(*dev_aval, host_aval, sizeof(valuetype) * nnz,
                        cudaMemcpyHostToDevice));
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <matrix file>\n", argv[0]);
    return 1;
  }

  srand(42);

  cusparseHandle_t handle;
  zero_check(cusparseCreate(&handle));

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

  int64_t *dev_acol, *dev_csr_arow;
  valuetype *dev_aval;
  allocate_device_buffers(host_csr_arow, host_acol, host_aval, rows, nnz,
                          &dev_acol, &dev_csr_arow, &dev_aval);

  TIMER_STOP(ParseTime);

  double parse_time = TIMER_ELAPSED(ParseTime) / 1e6;

  printf("parsed in %10fs\n", parse_time);

  valuetype *dev_dense_vec = init_dense_vec_dev(cols);
  valuetype *dev_out_coo;

  benchmark_results results[4];

  bench_expression(results[0] = benchmark_csr_cpu(host_csr_arow, host_acol,
                                                  host_aval, rows, cols, nnz,
                                                  &dev_out_coo, dev_dense_vec));
  bench_expression(results[1] = benchmark_csr_scalar(
                       dev_csr_arow, dev_acol, dev_aval, rows, nnz, dev_out_coo,
                       dev_dense_vec));
  bench_expression(results[2] = benchmark_csr_vector_warp(
                       dev_csr_arow, dev_acol, dev_aval, rows, nnz, dev_out_coo,
                       dev_dense_vec));
  bench_expression(results[3] = benchmark_cusparse_csr_gpu(
                       handle, dev_csr_arow, dev_acol, dev_aval, rows, cols,
                       nnz, dev_out_coo, dev_dense_vec));

  char output_file[256];
  char *job_id = getenv("SLURM_JOB_ID");
  if (job_id == NULL) {
    // good enough for local test runs
    snprintf(output_file, 256, "results/%d.csv", rand());
  } else {
    snprintf(output_file, 256, "results/%s.csv", job_id);
  }

  if (write_results(output_file, argv[1], results,
                    sizeof(results) / sizeof(benchmark_results)) != 0) {
    printf("there was an error writing results to file\n");
    return 3;
  }

  zero_check(cudaFree(dev_dense_vec));
  zero_check(cudaFree(dev_out_coo));

  zero_check(cudaFree(dev_acol));
  zero_check(cudaFree(dev_csr_arow));
  zero_check(cudaFree(dev_aval));

  free(host_csr_arow);
  free(host_acol);
  free(host_aval);

  zero_check(cusparseDestroy(handle));

  printf("done\n");

  return 0;
}
