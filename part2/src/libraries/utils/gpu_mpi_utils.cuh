#ifndef GPU_MPI_UTILS_H
#define GPU_MPI_UTILS_H

#include "constants.h"
#include "general_mpi_utils.h"
#include <stdio.h>

void exchange_columns_gpu(
    idxtype *d_editable_acol,
    idxtype local_nnz,
    idxtype total_dense_vec_size,
    const vtype *d_local_dense_vec,
    int local_dense_count,
    int rank,
    int comm_size,
    vtype **out_d_extended_dense_vec,
    int *out_extended_dense_vec_count,
    int *out_received_columns_count,
    ghost_exchange_metrics *metrics);

inline cudaDataType cusparse_vtype() {
  if (sizeof(vtype) == sizeof(float)) {
    return CUDA_R_32F;
  }

  if (sizeof(vtype) == sizeof(double)) {
    return CUDA_R_64F;
  }

  printf("unsupported vtype size for cuSPARSE\n");
  exit(1);
}

#endif
