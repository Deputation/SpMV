#ifndef MPI_CSR
#define MPI_CSR

#include "constants.h"
#include "mpi_benchmark_results.cuh"
#include "parallel_parse_file.h"

mpi_compute_benchmark_results compute_csr_cpu_and_validate(
    const char *file_path, header_info *header, const idxtype *csr_lrow,
    const idxtype *global_acol, const vtype *aval, idxtype rank_nnz,
    vtype **local_result_out, const vtype *rank_0_dense_vec, int rank,
    int comm_size);

mpi_compute_benchmark_results compute_csr_gpu_and_validate(
    header_info *header, const idxtype *csr_lrow, const idxtype *global_acol,
    const vtype *aval, idxtype rank_nnz, const vtype *local_cpu_result,
    const vtype *rank_0_dense_vec, int rank, int comm_size);

mpi_compute_benchmark_results compute_csr_cusparse_and_validate(
    header_info *header, const idxtype *csr_lrow, const idxtype *global_acol,
    const vtype *aval, idxtype rank_nnz, const vtype *local_cpu_result,
    const vtype *rank_0_dense_vec, int rank, int comm_size);

#endif
