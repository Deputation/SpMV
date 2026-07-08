#ifndef PARSE_FILE_H
#define PARSE_FILE_H

#include "constants.h"

#include <cstdint>

int read_characteristics(const char *file_name, int64_t *rows, int64_t *cols,
                         int64_t *nnz, bool* is_graph, bool* is_symmetric);

int parse_file_coo(const char *file_name, int64_t **arow, int64_t **acol,
                   valuetype **aval, int64_t *rows, int64_t *cols,
                   int64_t *nnz, bool* is_graph = NULL, bool* is_symmetric = NULL);

int parse_file_csr(const char *file_name, int64_t **csr_arow, int64_t **acol,
                   valuetype **aval, int64_t *rows, int64_t *cols,
                   int64_t *nnz, bool* is_graph = NULL, bool* is_symmetric = NULL);

#endif
