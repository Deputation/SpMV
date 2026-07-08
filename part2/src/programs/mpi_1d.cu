#include "host_mpi_utils.h"
#include "mpi_benchmark_results.cuh"
#include "mpi_coo.cuh"
#include "mpi_csr.cuh"
#include "parallel_parse_file.h"
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int device_count = 0;
  zero_check(cudaGetDeviceCount(&device_count));

  assert(device_count > 0);

  if (device_count == 1) {
  	printf("1 device found, setting device 0\n");
    zero_check(cudaSetDevice(0));
  } else {
    zero_check(cudaSetDevice(rank));
  }

  int current_device = 0;
  zero_check(cudaGetDevice(&current_device));

  cudaDeviceProp prop;
  zero_check(cudaGetDeviceProperties(&prop, current_device));

  printf("rank %d/%d using CUDA device %d: %s\n", rank, comm_size,
         current_device, prop.name);

  srand(42 + rank);

  if (argc != 2) {
    if (rank == 0) {
      printf("usage: %s <file_path>\n", argv[0]);
    }

    MPI_Abort(MPI_COMM_WORLD, 1);

    return 1;
  }

  mpi_general_benchmark_results benchmarks(argv[1]);

  vtype *rank_0_dense_vec = nullptr;

  header_info header;
  coo_entry *entries;
  idxtype rank_nnz;
  read_file(argv[1], &header, &entries, &rank_nnz, &benchmarks.parsing, rank,
            comm_size);

  metric_reduce_min_avg_max(rank_nnz, &benchmarks.parsing.rank_nnz.min,
                            &benchmarks.parsing.rank_nnz.avg,
                            &benchmarks.parsing.rank_nnz.max,
                            &benchmarks.parsing.rank_nnz.max_over_avg,
                            MPI_LONG_LONG, rank, comm_size);

  // coo evaluation scope
  {
    vtype *local_result_coo = nullptr;

    // the coo unpacking function which rewrites the coo_entry structures into
    // coo format does not edit the coo_entry array and treats it as const
    //
    // arow = global row representation (unused in kernels, used to compute
    // lrow) lrow = local row representation
    idxtype *arow, *lrow, *acol;
    vtype *aval;
    unpack_coo(entries, rank_nnz, &arow, &lrow, &acol, &aval);

    // this function will generate the dense_vec we will use
    // throughout the evaluation, it will also compute the vector against which
    // the mse will be computed for the GPU version
    benchmarks.results[0] = compute_coo_cpu_and_validate(
        argv[1], &header, lrow, acol, aval, rank_nnz, &local_result_coo,
        &rank_0_dense_vec, rank, comm_size);

    benchmarks.results[1] = compute_coo_gpu_and_validate(
        &header, lrow, acol, aval, rank_nnz, local_result_coo, rank_0_dense_vec,
        rank, comm_size);

    benchmarks.results[2] = compute_coo_cusparse_and_validate(
        &header, lrow, acol, aval, rank_nnz, local_result_coo, rank_0_dense_vec,
        rank, comm_size);

    free(arow);
    free(lrow);
    free(acol);
    free(aval);

    // we will compute another validation vector for csr using the same dense
    // vector
    free(local_result_coo);
  }

  // csr evaluation scope
  {
    vtype *local_result_csr = nullptr;

    // the csr version, however, sorts the entries and does not treat them as a
    // const array after computing the csr representation we therefore free the
    // entries array
    //
    // csr_lrow represents a compressed representation of the local row indices
    idxtype *csr_lrow, *acol;
    vtype *aval;
    int local_rows = compute_cyclic_count(header.rows, rank, comm_size);
    unpack_csr(entries, rank_nnz, local_rows, &csr_lrow, &acol, &aval);
    free(entries);

    benchmarks.results[3] = compute_csr_cpu_and_validate(
        argv[1], &header, csr_lrow, acol, aval, rank_nnz, &local_result_csr,
        rank_0_dense_vec, rank, comm_size);

    benchmarks.results[4] = compute_csr_gpu_and_validate(
        &header, csr_lrow, acol, aval, rank_nnz, local_result_csr,
        rank_0_dense_vec, rank, comm_size);

    benchmarks.results[5] = compute_csr_cusparse_and_validate(
        &header, csr_lrow, acol, aval, rank_nnz, local_result_csr,
        rank_0_dense_vec, rank, comm_size);

    free(csr_lrow);
    free(acol);
    free(aval);

    free(local_result_csr);
  }

  if (rank == 0) {
    benchmarks.print();

    if (benchmarks.write_csv(getenv("SLURM_JOB_ID"), rank, comm_size) != 0) {
      printf("error writing results to disk\n");
    }
  }

  MPI_Finalize();

  return 0;
}
