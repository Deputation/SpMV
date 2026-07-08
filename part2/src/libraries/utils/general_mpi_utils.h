#ifndef GENERAL_MPI_UTILS_H
#define GENERAL_MPI_UTILS_H

#include <cstdint>

struct ghost_exchange_metrics {
  double ghost_exchange_time;
  double ghost_exchange_alltoallv_time;
  uint64_t ghost_values_received;
  uint64_t ghost_bytes_sent;
  uint64_t ghost_bytes_received;
};

#endif
