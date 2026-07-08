#ifndef HOST_MPI_UTILS_H
#define HOST_MPI_UTILS_H

#include "general_mpi_utils.h"
#include "constants.h"
#include <mpi.h>

template <typename T>
static int compute_cyclic_count(T count, int rank, int comm_size) {
  if (static_cast<T>(rank) >= count) {
    return 0; // unlikely with current dataset
  }

  // amount of elements per rank
  return static_cast<int>(1 + (count - 1 - rank) / comm_size);
}

template <typename T>
static void cyclic_to_global(T *result, const T *cyclic,
                             const int *displacements, const int *counts,
                             int comm_size) {
  for (int i = 0; i < comm_size; i++) {
    for (int j = 0; j < counts[i]; j++) {
      // strided access with rank as base
      int index = i + j * comm_size;
      // access linearly the cyclic array and write back
      result[index] = cyclic[displacements[i] + j];
    }
  }
}

template <typename T>
static void global_to_cyclic(T *result, const T *global,
                             const int *displacements, const int *counts,
                             int comm_size) {
  for (int i = 0; i < comm_size; i++) {
    for (int j = 0; j < counts[i]; j++) {
      // opposite operation from above function
      int index = i + j * comm_size;

      result[displacements[i] + j] = global[index];
    }
  }
}

template <typename T>
void metric_reduce_min_avg_max(T metric, T *min_out, double *avg_out,
                               T *max_out, double *max_over_avg,
                               MPI_Datatype datatype, int rank, int comm_size) {
  T local_metric = metric;

  T min = 0;
  T max = 0;
  T sum = 0;

  MPI_Reduce(&local_metric, &min, 1, datatype, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_metric, &max, 1, datatype, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_metric, &sum, 1, datatype, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double avg = static_cast<double>(sum) / comm_size;

    *min_out = min;
    *avg_out = avg;
    *max_out = max;
    if (avg != 0.0) {
      *max_over_avg = static_cast<double>(max) / avg;
    } else {
      *max_over_avg = 0.0;
    }
  }
}

void exchange_columns(idxtype *editable_acol, idxtype local_nnz,
                      idxtype total_dense_vec_size,
                      const vtype *local_dense_vec, int local_dense_count,
                      int rank, int comm_size, vtype **out_extended_dense_vec,
                      int *out_extended_dense_vec_count,
                      int *out_received_columns_count,
                      ghost_exchange_metrics *metrics);
#endif
