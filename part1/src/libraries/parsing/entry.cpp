#include "entry.h"

int comp_entry(const void *left, const void *right) {
  if (((entry *)left)->row < ((entry *)right)->row) {
    return -1;
  }

  if (((entry *)left)->row > ((entry *)right)->row) {
    return 1;
  }

  if (((entry *)left)->col < ((entry *)right)->col) {
    return -1;
  }

  if (((entry *)left)->col > ((entry *)right)->col) {
    return 1;
  }

  return 0;
}
