#include "host_mpi_utils.h"
#include "parallel_parse_file.h"
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <string.h>

void exchange_columns(idxtype *editable_acol, idxtype local_nnz,
                      idxtype total_dense_vec_size,
                      const vtype *local_dense_vec, int local_dense_count,
                      int rank, int comm_size, vtype **out_extended_dense_vec,
                      int *out_extended_dense_vec_count,
                      int *out_received_columns_count,
                      ghost_exchange_metrics *metrics) {
  if (metrics != nullptr) {
    metrics->ghost_exchange_time = 0.0;
    metrics->ghost_exchange_alltoallv_time = 0.0;
    metrics->ghost_values_received = 0;
    metrics->ghost_bytes_sent = 0;
    metrics->ghost_bytes_received = 0;
  }

  double ghost_exchange_t0 = MPI_Wtime();

  int error = MPI_SUCCESS;

  int raw_cols_to_recv_count[comm_size];
  memset(raw_cols_to_recv_count, 0, sizeof(raw_cols_to_recv_count));

  for (idxtype i = 0; i < local_nnz; i++) {
    int owner = static_cast<int>(editable_acol[i] % comm_size);

    if (owner != rank) {
      raw_cols_to_recv_count[owner]++;
    }
  }

  int raw_cols_to_recv_displacements[comm_size];

  int raw_cols_to_recv_total = compute_displacements(
      raw_cols_to_recv_count, raw_cols_to_recv_displacements, comm_size);

  idxtype *raw_columns_needed = reinterpret_cast<idxtype *>(
      malloc(sizeof(idxtype) * raw_cols_to_recv_total));
  assert(raw_columns_needed != nullptr);
  memset(raw_columns_needed, 0, sizeof(idxtype) * raw_cols_to_recv_total);

  int raw_cols_to_recv_writing_position[comm_size];
  memcpy(raw_cols_to_recv_writing_position, raw_cols_to_recv_displacements,
         sizeof(raw_cols_to_recv_writing_position));

  for (idxtype i = 0; i < local_nnz; i++) {
    int owner = static_cast<int>(editable_acol[i] % comm_size);

    if (owner != rank) {
      int pos = raw_cols_to_recv_writing_position[owner]++;
      raw_columns_needed[pos] = editable_acol[i];
    }
  }

  for (int i = 0; i < comm_size; i++) {
    assert(raw_cols_to_recv_writing_position[i] ==
           raw_cols_to_recv_displacements[i] + raw_cols_to_recv_count[i]);
  }

  for (int i = 0; i < comm_size; i++) {
    int start = raw_cols_to_recv_displacements[i];
    int count = raw_cols_to_recv_count[i];

    std::sort(raw_columns_needed + start, raw_columns_needed + start + count);
  }

  int cols_to_recv_count[comm_size];
  memset(cols_to_recv_count, 0, sizeof(cols_to_recv_count));

  for (int i = 0; i < comm_size; i++) {
    int start = raw_cols_to_recv_displacements[i];
    int count = raw_cols_to_recv_count[i];

    for (int j = 0; j < count; j++) {
      if (j == 0 ||
          raw_columns_needed[start + j] != raw_columns_needed[start + j - 1]) {
        cols_to_recv_count[i]++;
      }
    }
  }

  int cols_to_recv_displacements[comm_size];

  int cols_to_recv_total = compute_displacements(
      cols_to_recv_count, cols_to_recv_displacements, comm_size);

  idxtype *columns_needed =
      reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * cols_to_recv_total));
  assert(columns_needed != nullptr);
  memset(columns_needed, 0, sizeof(idxtype) * cols_to_recv_total);

  for (int i = 0; i < comm_size; i++) {
    int raw_start = raw_cols_to_recv_displacements[i];
    int raw_count = raw_cols_to_recv_count[i];

    int out_start = cols_to_recv_displacements[i];
    int out = 0;

    for (int j = 0; j < raw_count; j++) {
      if (j == 0 || raw_columns_needed[raw_start + j] !=
                        raw_columns_needed[raw_start + j - 1]) {
        columns_needed[out_start + out] = raw_columns_needed[raw_start + j];
        out++;
      }
    }

    assert(out == cols_to_recv_count[i]);
  }

  free(raw_columns_needed);

  int cols_to_send_count[comm_size];
  memset(cols_to_send_count, 0, sizeof(cols_to_send_count));

  error = MPI_Alltoall(cols_to_recv_count, 1, MPI_INT, cols_to_send_count, 1,
                       MPI_INT, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  int cols_to_send_displacements[comm_size];

  int cols_to_send_total = compute_displacements(
      cols_to_send_count, cols_to_send_displacements, comm_size);

  uint64_t ghost_values_received = static_cast<uint64_t>(cols_to_recv_total);

  uint64_t ghost_bytes_sent = static_cast<uint64_t>(cols_to_recv_total) *
                                  static_cast<uint64_t>(sizeof(idxtype)) +
                              static_cast<uint64_t>(cols_to_send_total) *
                                  static_cast<uint64_t>(sizeof(vtype));

  uint64_t ghost_bytes_received = static_cast<uint64_t>(cols_to_send_total) *
                                      static_cast<uint64_t>(sizeof(idxtype)) +
                                  static_cast<uint64_t>(cols_to_recv_total) *
                                      static_cast<uint64_t>(sizeof(vtype));

  idxtype *columns_to_send =
      reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * cols_to_send_total));
  assert(columns_to_send != nullptr);
  memset(columns_to_send, 0, sizeof(idxtype) * cols_to_send_total);

  MPI_Datatype MPI_IDXTYPE;
  MPI_Type_contiguous(static_cast<int>(sizeof(idxtype)), MPI_BYTE,
                      &MPI_IDXTYPE);
  MPI_Type_commit(&MPI_IDXTYPE);

  double ghost_exchange_column_ids_alltoallv_t0 = MPI_Wtime();

  error = MPI_Alltoallv(
      columns_needed, cols_to_recv_count, cols_to_recv_displacements,
      MPI_IDXTYPE, columns_to_send, cols_to_send_count,
      cols_to_send_displacements, MPI_IDXTYPE, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  double ghost_exchange_column_ids_alltoallv_t1 = MPI_Wtime();

  MPI_Type_free(&MPI_IDXTYPE);

  vtype *cols_to_send =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * cols_to_send_total));
  assert(cols_to_send != nullptr);
  memset(cols_to_send, 0, sizeof(vtype) * cols_to_send_total);

  for (int i = 0; i < cols_to_send_total; i++) {
    idxtype col = columns_to_send[i];

    assert(col >= 0);
    assert(col < total_dense_vec_size);
    assert(static_cast<int>(col % comm_size) == rank);

    idxtype local_col_idx = col / comm_size;

    assert(local_col_idx >= 0);
    assert(local_col_idx < static_cast<idxtype>(local_dense_count));

    cols_to_send[i] = local_dense_vec[local_col_idx];
  }

  vtype *cols_to_recv =
      reinterpret_cast<vtype *>(malloc(sizeof(vtype) * cols_to_recv_total));
  assert(cols_to_recv != nullptr);
  memset(cols_to_recv, 0, sizeof(vtype) * cols_to_recv_total);

  double ghost_exchange_column_values_alltoallv_t0 = MPI_Wtime();

  error = MPI_Alltoallv(cols_to_send, cols_to_send_count,
                        cols_to_send_displacements, MPI_FLOAT, cols_to_recv,
                        cols_to_recv_count, cols_to_recv_displacements,
                        MPI_FLOAT, MPI_COMM_WORLD);
  assert(error == MPI_SUCCESS);

  double ghost_exchange_column_values_alltoallv_t1 = MPI_Wtime();

  int extended_dense_vec_count = local_dense_count + cols_to_recv_total;

  vtype *extended_dense_vec = reinterpret_cast<vtype *>(
      malloc(sizeof(vtype) * extended_dense_vec_count));
  assert(extended_dense_vec != nullptr);
  memset(extended_dense_vec, 0, sizeof(vtype) * extended_dense_vec_count);

  if (local_dense_count > 0) {
    memcpy(extended_dense_vec, local_dense_vec,
           sizeof(vtype) * local_dense_count);
  }

  if (cols_to_recv_total > 0) {
    memcpy(extended_dense_vec + local_dense_count, cols_to_recv,
           sizeof(vtype) * cols_to_recv_total);
  }

  for (idxtype i = 0; i < local_nnz; i++) {
    idxtype global_col = editable_acol[i];
    int owner = static_cast<int>(global_col % comm_size);

    if (owner == rank) {
      idxtype local_col_idx = global_col / comm_size;

      assert(local_col_idx >= 0);
      assert(local_col_idx < static_cast<idxtype>(local_dense_count));

      // acol[i] = local_col_idx;
      editable_acol[i] = local_col_idx;
    } else {
      idxtype *begin = columns_needed + cols_to_recv_displacements[owner];

      idxtype *end = begin + cols_to_recv_count[owner];

      idxtype *found = std::lower_bound(begin, end, global_col);

      assert(found != end);
      assert(*found == global_col);

      int ghost_slot = static_cast<int>(found - columns_needed);

      assert(ghost_slot >= 0);
      assert(ghost_slot < cols_to_recv_total);

      // acol[i] = static_cast<idxtype>(local_dense_count + ghost_slot);
      editable_acol[i] = static_cast<idxtype>(local_dense_count + ghost_slot);
    }

    assert(editable_acol[i] >= 0);
    assert(editable_acol[i] < static_cast<idxtype>(extended_dense_vec_count));
  }

  free(columns_needed);
  free(columns_to_send);
  free(cols_to_send);
  free(cols_to_recv);

  *out_extended_dense_vec = extended_dense_vec;
  *out_extended_dense_vec_count = extended_dense_vec_count;

  if (out_received_columns_count != nullptr) {
    *out_received_columns_count = cols_to_recv_total;
  }

  double ghost_exchange_t1 = MPI_Wtime();

  if (metrics != nullptr) {
    metrics->ghost_exchange_time = ghost_exchange_t1 - ghost_exchange_t0;
    // sum alltoallv calls for column ids and column values
    metrics->ghost_exchange_alltoallv_time =
        (ghost_exchange_column_ids_alltoallv_t1 -
         ghost_exchange_column_ids_alltoallv_t0) +
        (ghost_exchange_column_values_alltoallv_t1 -
         ghost_exchange_column_values_alltoallv_t0);
    metrics->ghost_values_received = ghost_values_received;
    metrics->ghost_bytes_sent = ghost_bytes_sent;
    metrics->ghost_bytes_received = ghost_bytes_received;
  }
}
