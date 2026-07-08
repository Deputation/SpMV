#ifndef ENTRY_H
#define ENTRY_H

#include "constants.h"

typedef struct {
  idxtype row;
  idxtype col;
  vtype val;
} entry;

int comp_entry(const void *left, const void *right);

#endif
