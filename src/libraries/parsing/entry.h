#ifndef ENTRY_H
#define ENTRY_H

#include "constants.h"
#include <cstdint>

typedef struct {
  int64_t row;
  int64_t col;
  valuetype val;
} entry;

int comp_entry(const void *left, const void *right);

#endif
