#ifndef PARALLEL_PARSE_FILE_H
#define PARALLEL_PARSE_FILE_H

#include <mpi.h>

#include "constants.h"
#include "mpi_benchmark_results.cuh"

#pragma pack(push, 1)
struct header_info {
  char file_path[256];
  idxtype rows;
  idxtype cols;
  idxtype pre_sym_nnz;
  MPI_Offset starting_offset;
  MPI_Offset file_size;
  bool graph;
  bool symmetric;
};

struct coo_entry {
  // local representation
  idxtype local_row;

  idxtype row;
  idxtype col;
  vtype val;

  coo_entry(idxtype row, idxtype col, vtype val)
      : local_row(0), row(row), col(col), val(val) {}
};
#pragma pack(pop)

void read_and_broadcast_header(int rank, const char *file_path,
                               header_info *header);

int compute_displacements(const int *entries, int *output, int size);

void parallel_read_coo(header_info *header, int rank, int comm_size,
                       coo_entry **entries, idxtype *out_local_nnz,
                       double *out_mpi_file_read_time,
                       double *out_alltoallv_entries_time,
                       uint64_t *out_entries_received,
                       uint64_t *out_entry_exchange_bytes_received);

void unpack_coo(const coo_entry *entries, idxtype local_nnz, idxtype **arow,
                idxtype **lrow, idxtype **acol, vtype **aval);

void unpack_csr(coo_entry *entries, idxtype local_nnz, int local_rows,
                idxtype **csr_lrow, idxtype **acol, vtype **aval);

void read_file(const char *file_path, header_info *header, coo_entry **entries,
               idxtype *rank_nnz,
               mpi_parsing_benchmark_results *parsing_results, int rank,
               int comm_size);

#endif
