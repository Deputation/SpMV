#ifndef PARSE_FILE_H
#define PARSE_FILE_H

#include "constants.h"

int read_characteristics(const char *file_name, idxtype *rows,
                         idxtype *cols, idxtype *nnz, bool *is_graph,
                         bool *is_symmetric);

int parse_file_coo(const char *file_name, idxtype **arow, idxtype **acol,
                   vtype **aval, idxtype *rows, idxtype *cols,
                   idxtype *nnz, bool *is_graph = NULL,
                   bool *is_symmetric = NULL);

int parse_file_csr(const char *file_name, idxtype **csr_arow,
                   idxtype **acol, vtype **aval, idxtype *rows,
                   idxtype *cols, idxtype *nnz, bool *is_graph = NULL,
                   bool *is_symmetric = NULL);

#endif
