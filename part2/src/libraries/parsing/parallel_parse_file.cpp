#include "parallel_parse_file.h"
#include "constants.h"
#include "mpi_benchmark_results.cuh"

#include <cassert>
#include <climits>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include <cstddef>
#include <cstdlib>
#include <unistd.h>

int gather_data(const char *file_path, header_info *output) {
  FILE *file = fopen(file_path, "r");
  assert(file != nullptr);

  bool graph = false;

  char first_line[128];
  if (fgets(first_line, 128, file) == NULL) {
    printf("error: %s %d %s\n", __FILE__, __LINE__, "could not read file");
    fclose(file);
    return 1;
  }

  char *first_tok = strtok(first_line, " ");
  if (strcmp(first_tok, "%%MatrixMarket") != 0) {
    printf("error reading first line\n");
    fclose(file);
    return 2;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "matrix") != 0) {
    printf("wrong type of data, matrix only\n");
    fclose(file);
    return 3;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "coordinate") != 0) {
    printf("wrong type of data, coordinate only\n");
    fclose(file);
    return 4;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "real") != 0) {
    if (strcmp(first_tok, "pattern") != 0) {
      printf("wrong type of data, real or pattern only\n");
      fclose(file);
      return 5;
    }

    // this is a graph
    graph = true;
    printf("graph detected\n");
  }

  bool symmetric = false;

  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "symmetric\n") != 0 &&
      strcmp(first_tok, "general\n") != 0) {
    printf("wrong type of matrix layout, symmetric or general only\n");
    fclose(file);
    return 6;
  }

  if (strcmp(first_tok, "symmetric\n") == 0) {
    symmetric = true;
    printf("symmetric matrix detected\n");
  }

  // skip comments
  char out[128];
  do {
    if (fgets(out, 128, file) == NULL) {
      printf("error: %s %d %s\n", __FILE__, __LINE__, "could not read file");
      fclose(file);
      return 7;
    }
  } while (out[0] == '%');

  // read initial information
  char *str = strtok(out, " ");

  idxtype rows = atol(str);
  idxtype cols = atol(strtok(NULL, " "));
  idxtype nnz = atol(strtok(NULL, " "));
  MPI_Offset starting_offset = static_cast<MPI_Offset>(ftello(file));

  if (fseeko(file, 0, SEEK_END) != 0) {
    printf("could not read file through to the end.\n");
    fclose(file);
    return 8;
  }

  MPI_Offset file_size = static_cast<MPI_Offset>(ftello(file));

  if (fclose(file) != 0) {
    printf("could not close file.\n");
    return 9;
  }

  output->rows = rows;
  output->cols = cols;
  // we will not double it just yet if it's symmetric as we can't know how many
  // elements are in the diagonal without reading the whole file
  output->pre_sym_nnz = nnz;
  output->starting_offset = starting_offset;
  output->file_size = file_size;
  output->symmetric = symmetric;
  output->graph = graph;

  return 0;
}

void print_header(header_info *header) {
  printf("rows: %ld, cols: %ld, nnz: %ld, graph: %s, symmetric: "
         "%s\nstarting_offset = %lld, file_size = %lld\n",
         header->rows, header->cols, header->pre_sym_nnz,
         header->graph ? "yes" : "no", header->symmetric ? "yes" : "no",
         header->starting_offset, header->file_size);
}

void read_and_broadcast_header(int rank, const char *file_path,
                               header_info *header) {
  if (rank == 0) {
    strcpy(header->file_path, file_path);
    int error = gather_data(file_path, header);

    if (error != 0) {
      MPI_Abort(MPI_COMM_WORLD, error);
    }

    // print_header(header);
  }

  MPI_Bcast(header, sizeof(*header), MPI_BYTE, 0, MPI_COMM_WORLD);

  assert(header->rows <= INT_MAX);
  assert(header->cols <= INT_MAX);
  assert(header->pre_sym_nnz <= INT_MAX);
}

MPI_Offset seek_line_start(MPI_File *file, MPI_Offset rank_start,
                           MPI_Offset starting_offset) {
  if (rank_start <= starting_offset) {
    return starting_offset;
  }

  char buf = 0;
  MPI_Status status;
  assert(MPI_File_read_at(*file, rank_start - 1, &buf, 1, MPI_CHAR, &status) ==
         MPI_SUCCESS);
  int count = 0;
  MPI_Get_count(&status, MPI_CHAR, &count);
  assert(count == 1);

  if (buf == '\n') {
    // we're already at the start of a line
    return rank_start;
  }

  while (buf != '\n') {
    rank_start--;

    if (rank_start <= starting_offset) {
      return starting_offset;
    }

    assert(MPI_File_read_at(*file, rank_start, &buf, 1, MPI_CHAR, &status) ==
           MPI_SUCCESS);

    count = 0;
    MPI_Get_count(&status, MPI_CHAR, &count);
    assert(count == 1);
  }

  rank_start++;

  return rank_start;
}

void count_lines(int comm_size, header_info *header, char *buffer,
                 MPI_Offset assigned_bytes, int *rank_counts) {
  char *pointer = buffer;
  char *buffer_end = buffer + assigned_bytes;

  while (pointer < buffer_end) {
    // scan for the newline
    char *newline =
        reinterpret_cast<char *>(memchr(pointer, '\n', buffer_end - pointer));
    char old_newline = '\x00';

    if (newline == nullptr) {
      // we're at buffer end
      newline = buffer_end;
    } else {
      old_newline = *newline;
      *newline = '\x00';
    }

    if (pointer[0] != '\x00') {
      char *str = pointer;
      char *next_str = nullptr;

      idxtype row = strtoll(str, &next_str, 10);
      assert(str != next_str);
      str = next_str;

      idxtype col = strtoll(str, &next_str, 10);
      assert(str != next_str);
      str = next_str;

      row -= 1;
      col -= 1;

      rank_counts[row % comm_size]++;

      if (header->symmetric && row != col) {
        // double increment, they will need the space, but use col as it will be
        // flipped
        rank_counts[col % comm_size]++;
      }
    }

    if (newline == buffer_end) {
      pointer = buffer_end;
    } else {
      pointer = newline + 1;
      *newline = old_newline;
    }
  }
}

int compute_displacements(const int *entries, int *output, int size) {
  output[0] = 0;

  for (int i = 1; i < size; i++) {
    output[i] = output[i - 1] + entries[i - 1];
  }

  return output[size - 1] + entries[size - 1];
}

void parallel_read_coo(header_info *header, int rank, int comm_size,
                       coo_entry **entries, idxtype *out_local_nnz,
                       double *out_mpi_file_read_time,
                       double *out_alltoallv_entries_time,
                       uint64_t *out_entries_received,
                       uint64_t *out_entry_exchange_bytes_received) {
  *out_mpi_file_read_time = 0.0;
  *out_alltoallv_entries_time = 0.0;
  *out_entries_received = 0;
  *out_entry_exchange_bytes_received = 0;

  // prepare to assign file chunks
  MPI_Offset casted_rank = static_cast<MPI_Offset>(rank);
  MPI_Offset total_data = (header->file_size - header->starting_offset);

  MPI_Offset rank_start =
      header->starting_offset + ((casted_rank * total_data) / comm_size);
  MPI_Offset rank_end =
      header->starting_offset + (((casted_rank + 1) * total_data) / comm_size);

  // printf("[%lld] rank_start: %lld ", casted_rank, rank_start);

  /*if (rank == (comm_size - 1)) {
    printf("rank_end: %lld ", rank_end);
    printf("file size: %lld\n", header->file_size);

    assert(rank_end == header->file_size);
  } else {
    printf("rank_end: %lld\n", rank_end);
    }*/

  // let's start reading
  MPI_File file;
  int err = MPI_File_open(MPI_COMM_WORLD, header->file_path, MPI_MODE_RDONLY,
                          MPI_INFO_NULL, &file);
  assert(err == MPI_SUCCESS);

  // align
  MPI_Offset line_start;
  if (rank == 0) {
    // we're already at a line start
    line_start = rank_start;
  } else {
    line_start = seek_line_start(&file, rank_start, header->starting_offset);
  }

  MPI_Offset line_end;
  if (rank == (comm_size - 1)) {
    line_end = header->file_size;
  } else {
    line_end = seek_line_start(&file, rank_end, header->starting_offset);
  }

  // prepare to read aligned chunk
  MPI_Offset assigned_bytes = line_end - line_start;
  assert(assigned_bytes <= INT_MAX);

  assert(assigned_bytes != 0);
  char *text_data = reinterpret_cast<char *>(malloc(assigned_bytes));
  assert(text_data != nullptr);

  // read
  MPI_Status status;

  double mpi_read_t0 = MPI_Wtime();

  int error = MPI_File_read_at_all(
      file, line_start, reinterpret_cast<void *>(text_data),
      static_cast<int>(assigned_bytes), MPI_CHAR, &status);
  assert(error == MPI_SUCCESS);

  double mpi_read_t1 = MPI_Wtime();

  *out_mpi_file_read_time = mpi_read_t1 - mpi_read_t0;

  int count = 0;
  MPI_Get_count(&status, MPI_CHAR, &count);
  assert(static_cast<MPI_Offset>(count) == assigned_bytes);

  // some of the next arrays will be made to be integers because of the future
  // MPI_Alltoallv call

  // tells us how much data every rank will have to store
  // rank_counts[0] -> data this rank will send to rank 0
  // rank_counts[1] -> data this rank will send to rank 1
  // ...
  int sending_rank_counts[comm_size];
  memset(sending_rank_counts, 0, sizeof(sending_rank_counts));
  count_lines(comm_size, header, text_data, assigned_bytes,
              sending_rank_counts);

  // exchange counts among ranks
  int receiving_rank_counts[comm_size];
  MPI_Alltoall(sending_rank_counts, 1, MPI_INT, receiving_rank_counts, 1,
               MPI_INT, MPI_COMM_WORLD);

  // receiving_rank_counts[rank] -> what we will parse for ourselves
  // receiving_rank_counts[other] -> what other has to send us

  int send_displacements[comm_size];
  int receive_displacements[comm_size];

  int total_to_send =
      compute_displacements(sending_rank_counts, send_displacements, comm_size);

  int total_to_receive = compute_displacements(
      receiving_rank_counts, receive_displacements, comm_size);

  // allocate sending buffers
  coo_entry *sending_entries =
      reinterpret_cast<coo_entry *>(malloc(sizeof(coo_entry) * total_to_send));
  int sending_entries_writing_idx[comm_size];
  memcpy(sending_entries_writing_idx, send_displacements,
         sizeof(int) * comm_size);

  // allocate receiving buffers
  coo_entry *receiving_entries = reinterpret_cast<coo_entry *>(
      malloc(sizeof(coo_entry) * total_to_receive));

  assert(sending_entries != nullptr || total_to_send == 0);
  assert(receiving_entries != nullptr || total_to_receive == 0);

  char *pointer = text_data;
  char *buffer_end = text_data + assigned_bytes;

  while (pointer < buffer_end) {
    // scan for the newline
    char *newline =
        reinterpret_cast<char *>(memchr(pointer, '\n', buffer_end - pointer));
    char old_newline = '\x00';

    if (newline == nullptr) {
      // we're at buffer end
      newline = buffer_end;
    } else {
      old_newline = *newline;
      *newline = '\x00';
    }

    if (pointer[0] != '\x00') {
      char *str = pointer;
      char *next_str = nullptr;

      idxtype row = strtoll(str, &next_str, 10);
      assert(str != next_str);
      str = next_str;

      idxtype col = strtoll(str, &next_str, 10);
      assert(str != next_str);
      str = next_str;

      vtype val = 1.0;
      if (!header->graph) {
        val = strtof(str, &next_str);
        assert(str != next_str);
        str = next_str;
      }

      row -= 1;
      col -= 1;

      // we have to write this entry to the send buffer, and write another entry
      // if it's symmetric AND NOT diagonal
      sending_entries[sending_entries_writing_idx[row % comm_size]++] =
          coo_entry(row, col, val);

      if (header->symmetric && row != col) {
        sending_entries[sending_entries_writing_idx[col % comm_size]++] =
            coo_entry(col, row, val);
      }
    }

    if (newline != buffer_end) {
      *newline = old_newline;
      pointer = newline + 1;
    } else {
      pointer = buffer_end;
    }
  }

  for (int i = 0; i < comm_size; i++) {
    assert(sending_entries_writing_idx[i] ==
           send_displacements[i] + sending_rank_counts[i]);
  }

  free(text_data);
  MPI_File_close(&file);

  MPI_Datatype MPI_COO_ENTRY;
  MPI_Type_contiguous(sizeof(coo_entry), MPI_BYTE, &MPI_COO_ENTRY);
  MPI_Type_commit(&MPI_COO_ENTRY);

  double alltoallv_t0 = MPI_Wtime();

  MPI_Alltoallv(sending_entries, sending_rank_counts, send_displacements,
                MPI_COO_ENTRY, receiving_entries, receiving_rank_counts,
                receive_displacements, MPI_COO_ENTRY, MPI_COMM_WORLD);

  double alltoallv_t1 = MPI_Wtime();

  *out_alltoallv_entries_time = alltoallv_t1 - alltoallv_t0;

  MPI_Type_free(&MPI_COO_ENTRY);

  idxtype rank_nnz = total_to_receive;

  for (idxtype i = 0; i < rank_nnz; i++) {
    assert(receiving_entries[i].row % comm_size == rank);
    // local representation
    receiving_entries[i].local_row = receiving_entries[i].row / comm_size;
  }

  *out_entries_received = static_cast<uint64_t>(total_to_receive);
  *out_entry_exchange_bytes_received = static_cast<uint64_t>(total_to_receive) *
                                       static_cast<uint64_t>(sizeof(coo_entry));

  *entries = receiving_entries;
  *out_local_nnz = rank_nnz;

  free(sending_entries);
}

void unpack_coo(const coo_entry *entries, idxtype local_nnz, idxtype **arow,
                idxtype **lrow, idxtype **acol, vtype **aval) {
  assert(local_nnz != 0);
  *arow = reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * local_nnz));
  assert((*arow) != nullptr);
  *lrow = reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * local_nnz));
  assert(*(lrow) != nullptr);
  *acol = reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * local_nnz));
  assert(*(acol) != nullptr);
  *aval = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_nnz));
  assert(*(aval) != nullptr);
  for (idxtype i = 0; i < local_nnz; i++) {
    (*arow)[i] = entries[i].row;
    (*lrow)[i] = entries[i].local_row;
    (*acol)[i] = entries[i].col;
    (*aval)[i] = entries[i].val;
  }
}

int comp_coo_entry_local_row_col(const void *left, const void *right) {
  const coo_entry *l = reinterpret_cast<const coo_entry *>(left);
  const coo_entry *r = reinterpret_cast<const coo_entry *>(right);

  if (l->local_row < r->local_row) {
    return -1;
  }

  if (l->local_row > r->local_row) {
    return 1;
  }

  if (l->col < r->col) {
    return -1;
  }

  if (l->col > r->col) {
    return 1;
  }

  return 0;
}

void unpack_csr(coo_entry *entries, idxtype local_nnz, int local_rows,
                idxtype **csr_lrow, idxtype **acol, vtype **aval) {
  int row_ptr_size = local_rows + 1;

  *csr_lrow =
      reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * row_ptr_size));
  assert((*csr_lrow) != nullptr);
  memset(*csr_lrow, 0, sizeof(idxtype) * row_ptr_size);

  *acol = reinterpret_cast<idxtype *>(malloc(sizeof(idxtype) * local_nnz));
  assert((*acol) != nullptr);

  *aval = reinterpret_cast<vtype *>(malloc(sizeof(vtype) * local_nnz));
  assert((*aval) != nullptr);

  qsort(entries, static_cast<size_t>(local_nnz), sizeof(coo_entry),
        comp_coo_entry_local_row_col);

  for (idxtype i = 0; i < local_nnz; i++) {
    idxtype row = entries[i].local_row;
    (*csr_lrow)[row + 1]++;
  }

  for (int r = 0; r < local_rows; r++) {
    (*csr_lrow)[r + 1] += (*csr_lrow)[r];
  }

  assert((*csr_lrow)[0] == 0);
  assert((*csr_lrow)[local_rows] == local_nnz);

  for (idxtype i = 0; i < local_nnz; i++) {
    (*acol)[i] = entries[i].col;
    (*aval)[i] = entries[i].val;
  }
}

void read_file(const char *file_path, header_info *header, coo_entry **entries,
               idxtype *rank_nnz,
               mpi_parsing_benchmark_results *parsing_results, int rank,
               int comm_size) {
  double mpi_file_read_time = 0.0;
  double alltoallv_entries_time = 0.0;

  uint64_t entries_received = 0;
  uint64_t entry_exchange_bytes_received = 0;

  double total_t0 = MPI_Wtime();

  read_and_broadcast_header(rank, file_path, header);

  parallel_read_coo(header, rank, comm_size, entries, rank_nnz,
                    &mpi_file_read_time, &alltoallv_entries_time,
                    &entries_received, &entry_exchange_bytes_received);

  double total_t1 = MPI_Wtime();

  double total_time = total_t1 - total_t0;

  parsing_results->populate(total_time, mpi_file_read_time,
                            alltoallv_entries_time, *rank_nnz, entries_received,
                            entry_exchange_bytes_received, rank, comm_size);
}
