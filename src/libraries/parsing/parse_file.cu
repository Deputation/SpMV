#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "entry.h"
#include "constants.h"

int read_characteristics(const char *file_name, int64_t *rows, int64_t *cols,
                         int64_t *nnz, bool *is_graph, bool *is_symmetric) {
  FILE *file = fopen(file_name, "r");
  bool graph = false;

  char first_line[128];
  if (fgets(first_line, 128, file) == NULL) {
    printf("error: %s %d %s\n", __FILE__, __LINE__, "could not read file");
    exit(1);
  }

  char *first_tok = strtok(first_line, " ");
  if (strcmp(first_tok, "%%MatrixMarket") != 0) {
    printf("error reading first line\n");
    return 2;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "matrix") != 0) {
    printf("wrong type of data, matrix only\n");
    return 3;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "coordinate") != 0) {
    printf("wrong type of data, coordinate only\n");
    return 4;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "real") != 0) {
    if (strcmp(first_tok, "pattern") != 0) {
      printf("wrong type of data, real or pattern only\n");
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
      exit(1);
    }
  } while (out[0] == '%');

  // read initial information
  char *str = strtok(out, " ");

  int64_t inner_rows = atol(str);
  int64_t inner_cols = atol(strtok(NULL, " "));
  int64_t inner_nnz = atol(strtok(NULL, " "));

  if (symmetric) {
    inner_nnz *= 2;
  }

  *rows = inner_rows;
  *cols = inner_cols;
  *nnz = inner_nnz;
  *is_graph = graph;
  *is_symmetric = symmetric;

  return 0;
}

int parse_file_coo(const char *file_name, int64_t **arow, int64_t **acol,
                   valuetype **aval, int64_t *rows, int64_t *cols, int64_t *nnz,
                   bool *is_graph, bool *is_symmetric) {
  FILE *file = fopen(file_name, "r");
  bool graph = false;

  char first_line[128];
  if (fgets(first_line, 128, file) == NULL) {
    printf("error: %s %d %s\n", __FILE__, __LINE__, "could not read file");
    exit(1);
  }

  char *first_tok = strtok(first_line, " ");
  if (strcmp(first_tok, "%%MatrixMarket") != 0) {
    printf("error reading first line\n");
    return 2;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "matrix") != 0) {
    printf("wrong type of data, matrix only\n");
    return 3;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "coordinate") != 0) {
    printf("wrong type of data, coordinate only\n");
    return 4;
  }
  first_tok = strtok(NULL, " ");
  if (strcmp(first_tok, "real") != 0) {
    if (strcmp(first_tok, "pattern") != 0) {
      printf("wrong type of data, real or pattern only\n");
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
      exit(1);
    }
  } while (out[0] == '%');

  // read initial information
  char *str = strtok(out, " ");

  int64_t inner_rows = atol(str);
  int64_t inner_cols = atol(strtok(NULL, " "));
  int64_t inner_nnz = atol(strtok(NULL, " "));

  if (symmetric) {
    inner_nnz *= 2;
  }

  int64_t *inner_arow = (int64_t *)malloc(sizeof(int64_t) * inner_nnz);
  int64_t *inner_acol = (int64_t *)malloc(sizeof(int64_t) * inner_nnz);
  valuetype *inner_aval = (valuetype *)malloc(sizeof(valuetype) * inner_nnz);

  assert(inner_arow != nullptr);
  assert(inner_acol != nullptr);
  assert(inner_aval != nullptr);

  int64_t last_index = 0;
  for (int64_t i = 0; fgets(out, 128, file) != NULL; i++) {
    char *str = strtok(out, " ");
    int64_t row = atol(str);
    int64_t col = atol(strtok(NULL, " "));

    // these files are not 0 indexed
    row -= 1;
    col -= 1;

    inner_arow[i] = row;
    inner_acol[i] = col;

    if (graph) {
      inner_aval[i] = 1;
    } else {
      valuetype val = atof(strtok(NULL, " "));
      inner_aval[i] = val;
    }

    last_index = i;
  }

  if (symmetric) {
    printf("replicating symmetry\n");
    last_index++;

    int symmetry_index = last_index;
    for (int64_t i = 0; i < (inner_nnz / 2); i++) {
      int64_t row_val = inner_arow[i];
      int64_t col_val = inner_acol[i];

      if (row_val == col_val) {
        continue;
      }

      // mirror
      inner_arow[symmetry_index] = col_val;
      inner_acol[symmetry_index] = row_val;
      inner_aval[symmetry_index] = inner_aval[i];
      symmetry_index++;
    }

    inner_nnz = symmetry_index;

    int64_t *correct_inner_arow =
        (int64_t *)malloc(sizeof(int64_t) * inner_nnz);
    int64_t *correct_inner_acol =
        (int64_t *)malloc(sizeof(int64_t) * inner_nnz);
    valuetype *correct_inner_aval =
        (valuetype *)malloc(sizeof(valuetype) * inner_nnz);

    assert(correct_inner_arow != nullptr);
    assert(correct_inner_acol != nullptr);
    assert(correct_inner_aval != nullptr);

    memcpy(correct_inner_arow, inner_arow, sizeof(int64_t) * inner_nnz);
    memcpy(correct_inner_acol, inner_acol, sizeof(int64_t) * inner_nnz);
    memcpy(correct_inner_aval, inner_aval, sizeof(valuetype) * inner_nnz);

    free(inner_arow);
    free(inner_acol);
    free(inner_aval);

    inner_arow = correct_inner_arow;
    inner_acol = correct_inner_acol;
    inner_aval = correct_inner_aval;
  }

  *arow = inner_arow;
  *acol = inner_acol;
  *aval = inner_aval;
  *rows = inner_rows;
  *cols = inner_cols;
  *nnz = inner_nnz;

  if (is_graph) {
    *is_graph = graph;
  }

  if (is_symmetric) {
    *is_symmetric = symmetric;
  }

  fclose(file);

  return 0;
}

int parse_file_csr(const char *file_name, int64_t **csr_arow, int64_t **acol,
                   valuetype **aval, int64_t *rows, int64_t *cols, int64_t *nnz,
                   bool *is_graph, bool *is_symmetric) {
  int64_t *arow;
  if (parse_file_coo(file_name, &arow, acol, aval, rows, cols, nnz, is_graph,
                     is_symmetric) != 0) {
    printf("error encountered when parsing file\n");
    return 2;
  }

  assert(nnz != nullptr);
  assert((*nnz) != 0);

  entry *entries = (entry *)malloc(sizeof(entry) * (*nnz));
  assert(entries != nullptr);

  for (int64_t i = 0; i < (*nnz); i++) {
    entries[i].row = arow[i];
    entries[i].col = (*acol)[i];
    entries[i].val = (*aval)[i];
  }

  qsort(entries, *nnz, sizeof(entry), comp_entry);

  // updated with sorted values
  for (int64_t i = 0; i < (*nnz); i++) {
    arow[i] = entries[i].row;
    (*acol)[i] = entries[i].col;
    (*aval)[i] = entries[i].val;
  }

  // compute the # of elements
  int64_t *row_amounts = (int64_t *)calloc((*rows), sizeof(int64_t));
  assert(row_amounts != nullptr);

  for (int64_t i = 0; i < (*nnz); i++) {
    row_amounts[arow[i]]++;
  }

  // prefix sum
  int64_t *csr_arow_output = (int64_t *)malloc(sizeof(int64_t) * ((*rows) + 1));
  assert(csr_arow_output != nullptr);

  csr_arow_output[0] = 0;
  for (int64_t i = 1; i <= (*rows); i++) {
    csr_arow_output[i] = row_amounts[i - 1] + csr_arow_output[i - 1];
  }

  *csr_arow = csr_arow_output;

  free(arow);
  free(entries);
  free(row_amounts);

  return 0;
}
