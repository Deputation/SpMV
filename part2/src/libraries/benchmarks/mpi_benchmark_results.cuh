#ifndef MPI_BENCHMARK_RESULTS_H
#define MPI_BENCHMARK_RESULTS_H

#include "constants.h"
#include "host_mpi_utils.h"
#include "host_utils.cuh"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

template <typename T> struct min_max_avg {
  T min;
  double avg;
  T max;
  double max_over_avg;

  min_max_avg() : min(0), avg(0), max(0), max_over_avg(0) {}

  void print(const char *name) {
    if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
      printf("%s: min: %lld, avg: %.10f, max: %lld, max / avg: %.10f\n", name,
             static_cast<long long>(min), avg, static_cast<long long>(max),
             max_over_avg);
    } else if constexpr (std::is_integral<T>::value &&
                         std::is_unsigned<T>::value) {
      printf("%s: min: %llu, avg: %.10f, max: %llu, max / avg: %.10f\n", name,
             static_cast<unsigned long long>(min), avg,
             static_cast<unsigned long long>(max), max_over_avg);
    } else if constexpr (std::is_floating_point<T>::value) {
      printf("%s: min: %.10f, avg: %.10f, max: %.10f, max / avg: %.10f\n", name,
             static_cast<double>(min), avg, static_cast<double>(max),
             max_over_avg);
    }
  }

  void write_csv(FILE *f) const {
    if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
      fprintf(f, ",%lld,%.10f,%lld,%.10f", static_cast<long long>(min), avg,
              static_cast<long long>(max), max_over_avg);
    } else if constexpr (std::is_integral<T>::value &&
                         std::is_unsigned<T>::value) {
      fprintf(f, ",%llu,%.10f,%llu,%.10f", static_cast<unsigned long long>(min),
              avg, static_cast<unsigned long long>(max), max_over_avg);
    } else if constexpr (std::is_floating_point<T>::value) {
      fprintf(f, ",%.10f,%.10f,%.10f,%.10f", static_cast<double>(min), avg,
              static_cast<double>(max), max_over_avg);
    }
  }
};

struct mpi_parsing_local_metrics {
  double total_time;
  double mpi_file_read_time;
  double alltoallv_entries_time;

  uint64_t rank_nnz;
  uint64_t entries_received;
  uint64_t entry_exchange_bytes_received;

  mpi_parsing_local_metrics() { clear(); }

  void clear() { memset(this, 0, sizeof(*this)); }
};

struct mpi_parsing_benchmark_results {
  min_max_avg<idxtype> rank_nnz;

  min_max_avg<double> total_time;
  min_max_avg<double> mpi_file_read_time;
  min_max_avg<double> alltoallv_entries_time;

  min_max_avg<uint64_t> entries_received;
  min_max_avg<uint64_t> entry_exchange_bytes_received;

  void populate(double local_total_time, double local_mpi_file_read_time,
                double local_alltoallv_entries_time, idxtype local_rank_nnz,
                uint64_t local_entries_received,
                uint64_t local_entry_exchange_bytes_received, int rank,
                int comm_size) {
    metric_reduce_min_avg_max(local_rank_nnz, &rank_nnz.min, &rank_nnz.avg,
                              &rank_nnz.max, &rank_nnz.max_over_avg,
                              MPI_INT64_T, rank, comm_size);

    metric_reduce_min_avg_max(
        local_total_time, &total_time.min, &total_time.avg, &total_time.max,
        &total_time.max_over_avg, MPI_DOUBLE, rank, comm_size);

    metric_reduce_min_avg_max(local_mpi_file_read_time, &mpi_file_read_time.min,
                              &mpi_file_read_time.avg, &mpi_file_read_time.max,
                              &mpi_file_read_time.max_over_avg, MPI_DOUBLE,
                              rank, comm_size);

    metric_reduce_min_avg_max(
        local_alltoallv_entries_time, &alltoallv_entries_time.min,
        &alltoallv_entries_time.avg, &alltoallv_entries_time.max,
        &alltoallv_entries_time.max_over_avg, MPI_DOUBLE, rank, comm_size);

    metric_reduce_min_avg_max(local_entries_received, &entries_received.min,
                              &entries_received.avg, &entries_received.max,
                              &entries_received.max_over_avg, MPI_UINT64_T,
                              rank, comm_size);

    metric_reduce_min_avg_max(
        local_entry_exchange_bytes_received, &entry_exchange_bytes_received.min,
        &entry_exchange_bytes_received.avg, &entry_exchange_bytes_received.max,
        &entry_exchange_bytes_received.max_over_avg, MPI_UINT64_T, rank,
        comm_size);
  }

  void print() {
    printf("Parsing results:\n");
    rank_nnz.print("rank_nnz");
    total_time.print("total_time");
    mpi_file_read_time.print("mpi_file_read_time");
    alltoallv_entries_time.print("alltoallv_entries_time");
    entries_received.print("entries_received");
    entry_exchange_bytes_received.print("entry_exchange_bytes_received");
  }
};

struct mpi_compute_benchmark_results {
  char result_type[256];

  uint64_t total_flops;
  min_max_avg<uint64_t> rank_flops;

  double global_kernel_gflops;
  double global_kernel_plus_full_ghost_gflops;
  double global_kernel_plus_alltoallv_ghost_gflops;

  min_max_avg<double> ghost_exchange_time;
  min_max_avg<double> ghost_exchange_alltoallv_time;
  min_max_avg<double> ghost_exchange_time_mean;
  min_max_avg<double> ghost_exchange_alltoallv_time_mean;
  min_max_avg<uint64_t> ghost_values_received;
  min_max_avg<uint64_t> ghost_bytes_sent;
  min_max_avg<uint64_t> ghost_bytes_received;

  min_max_avg<double> kernel_time;
  min_max_avg<double> kernel_time_mean;

  double error;

  mpi_compute_benchmark_results()
      : total_flops(0), global_kernel_gflops(0.0),
        global_kernel_plus_full_ghost_gflops(0.0),
        global_kernel_plus_alltoallv_ghost_gflops(0.0), error(0.0) {
    strcpy(result_type, "");
  }
  mpi_compute_benchmark_results(const char *type)
      : total_flops(0), global_kernel_gflops(0.0),
        global_kernel_plus_full_ghost_gflops(0.0),
        global_kernel_plus_alltoallv_ghost_gflops(0.0), error(0.0) {
    strcpy(result_type, type);
  }

  void populate_flop_metrics(idxtype local_nnz, int rank, int comm_size) {
    // nnz * 2
    uint64_t local_flops = static_cast<uint64_t>(local_nnz) * 2ULL;

    metric_reduce_min_avg_max(local_flops, &rank_flops.min, &rank_flops.avg,
                              &rank_flops.max, &rank_flops.max_over_avg,
                              MPI_UINT64_T, rank, comm_size);

    uint64_t global_flops = 0;

    MPI_Reduce(&local_flops, &global_flops, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      total_flops = global_flops;
      global_kernel_gflops = 0.0;
      global_kernel_plus_full_ghost_gflops = 0.0;
      global_kernel_plus_alltoallv_ghost_gflops = 0.0;

      if (kernel_time_mean.max > 0.0) {
        // aggregate kernel throughput, the distributed kernel completes when
        // the slowest rank completes, so total work is divided by the maximum
        // mean kernel time across ranks
        global_kernel_gflops =
            static_cast<double>(total_flops) / kernel_time_mean.max / 1e9;
      }

      double kernel_plus_ghost_time =
          kernel_time_mean.max + ghost_exchange_time_mean.max;

      if (kernel_plus_ghost_time > 0.0) {
        // communication inclusive aggregate throughput, this approximates one
        // distributed SpMV step as ghost-vector exchange plus local kernel
        // execution
        global_kernel_plus_full_ghost_gflops =
            static_cast<double>(total_flops) / kernel_plus_ghost_time / 1e9;
      }

      double kernel_plus_ghost_alltoallv_time =
          kernel_time_mean.max + ghost_exchange_alltoallv_time_mean.max;

      if (kernel_plus_ghost_alltoallv_time > 0.0) {
        // here we only take into account the overhead of the main alltoallv
        // exchange
        global_kernel_plus_alltoallv_ghost_gflops =
            static_cast<double>(total_flops) /
            kernel_plus_ghost_alltoallv_time / 1e9;
      }
    }
  }

  void populate_ghost_exchange_metrics(
      double *ghost_exchange_times, double *ghost_exchange_alltoallv_times,
      size_t length, double first_run_time, double first_run_alltoallv_time,
      uint64_t local_ghost_values_received, uint64_t local_ghost_bytes_sent,
      uint64_t local_ghost_bytes_received, int rank, int comm_size) {
    double mean_ghost_exchange_time =
        arithmetic_mean(ghost_exchange_times, length);
    double mean_ghost_exchange_alltoallv_time =
        arithmetic_mean(ghost_exchange_alltoallv_times, length);

    metric_reduce_min_avg_max(
        first_run_time, &ghost_exchange_time.min, &ghost_exchange_time.avg,
        &ghost_exchange_time.max, &ghost_exchange_time.max_over_avg, MPI_DOUBLE,
        rank, comm_size);

    metric_reduce_min_avg_max(
        first_run_alltoallv_time, &ghost_exchange_alltoallv_time.min,
        &ghost_exchange_alltoallv_time.avg, &ghost_exchange_alltoallv_time.max,
        &ghost_exchange_alltoallv_time.max_over_avg, MPI_DOUBLE, rank,
        comm_size);

    metric_reduce_min_avg_max(
        mean_ghost_exchange_time, &ghost_exchange_time_mean.min,
        &ghost_exchange_time_mean.avg, &ghost_exchange_time_mean.max,
        &ghost_exchange_time_mean.max_over_avg, MPI_DOUBLE, rank, comm_size);

    metric_reduce_min_avg_max(mean_ghost_exchange_alltoallv_time,
                              &ghost_exchange_alltoallv_time_mean.min,
                              &ghost_exchange_alltoallv_time_mean.avg,
                              &ghost_exchange_alltoallv_time_mean.max,
                              &ghost_exchange_alltoallv_time_mean.max_over_avg,
                              MPI_DOUBLE, rank, comm_size);

    metric_reduce_min_avg_max(
        local_ghost_values_received, &ghost_values_received.min,
        &ghost_values_received.avg, &ghost_values_received.max,
        &ghost_values_received.max_over_avg, MPI_UINT64_T, rank, comm_size);

    metric_reduce_min_avg_max(local_ghost_bytes_sent, &ghost_bytes_sent.min,
                              &ghost_bytes_sent.avg, &ghost_bytes_sent.max,
                              &ghost_bytes_sent.max_over_avg, MPI_UINT64_T,
                              rank, comm_size);

    metric_reduce_min_avg_max(
        local_ghost_bytes_received, &ghost_bytes_received.min,
        &ghost_bytes_received.avg, &ghost_bytes_received.max,
        &ghost_bytes_received.max_over_avg, MPI_UINT64_T, rank, comm_size);
  }

  void populate_kernel_times(double *kernel_times, size_t length,
                             double first_run_time, int rank, int comm_size) {
    double mean_kernel_time = arithmetic_mean(kernel_times, length);

    metric_reduce_min_avg_max(
        first_run_time, &kernel_time.min, &kernel_time.avg, &kernel_time.max,
        &kernel_time.max_over_avg, MPI_DOUBLE, rank, comm_size);

    metric_reduce_min_avg_max(mean_kernel_time, &kernel_time_mean.min,
                              &kernel_time_mean.avg, &kernel_time_mean.max,
                              &kernel_time_mean.max_over_avg, MPI_DOUBLE, rank,
                              comm_size);
  }

  void print() {
    printf("Result type: %s\n", result_type);

    printf("Total FLOPs: %lu\n", total_flops);

    rank_flops.print("FLOPs per rank");

    printf("Kernel GFLOP/s: %.10f\n", global_kernel_gflops);
    printf("Kernel + full ghost exchange GFLOP/s: %.10f\n",
           global_kernel_plus_full_ghost_gflops);
    printf("Kernel + ghost exchange alltoallv time GFLOP/s: %.10f\n",
           global_kernel_plus_alltoallv_ghost_gflops);

    ghost_exchange_time.print("ghost_exchange_time");
    ghost_exchange_time_mean.print("ghost_exchange_time_mean");

    ghost_exchange_alltoallv_time.print("ghost_exchange_alltoallv_time");
    ghost_exchange_alltoallv_time_mean.print(
        "ghost_exchange_alltoallv_time_mean");

    ghost_values_received.print("ghost_values_received");
    ghost_bytes_sent.print("ghost_bytes_sent");
    ghost_bytes_received.print("ghost_bytes_received");
    kernel_time.print("kernel_time");
    kernel_time_mean.print("kernel_time_mean");
  }
};

struct mpi_general_benchmark_results {
  char matrix[256];
  mpi_parsing_benchmark_results parsing;

  // coo, csr, coo gpu, csr gpu
  mpi_compute_benchmark_results results[6];

  mpi_general_benchmark_results(const char *matrix_name) {
    strcpy(matrix, matrix_name);
  }

  void print() {
    printf("---\n");
    printf("%s\n", matrix);
    parsing.print();
    for (size_t i = 0;
         i < sizeof(results) / sizeof(mpi_compute_benchmark_results); i++) {
      printf("---\n");
      results[i].print();
    }
    printf("---\n");
  }

  int write_csv(const char *slurm_job_id_env_var, int rank,
                int comm_size) const {
    if (rank != 0) {
      return 0;
    }

    char run_id[64];

    if (slurm_job_id_env_var != nullptr && slurm_job_id_env_var[0] != '\0') {
      snprintf(run_id, sizeof(run_id), "%s", slurm_job_id_env_var);
    } else {
      snprintf(run_id, sizeof(run_id), "%d", rand());
    }

    char parsing_name[512];
    char compute_name[512];

    snprintf(parsing_name, sizeof(parsing_name), "results/%d_%s_parsing.csv",
             comm_size, run_id);
    snprintf(compute_name, sizeof(compute_name), "results/%d_%s_compute.csv",
             comm_size, run_id);

    bool parsing_exists = access(parsing_name, F_OK) == 0;
    FILE *parsing_file = fopen(parsing_name, parsing_exists ? "a" : "w");

    if (parsing_file == nullptr) {
      return 1;
    }

    if (!parsing_exists) {
      fprintf(parsing_file,
              "run_id,nprocs,matrix,"
              "rank_nnz_min,rank_nnz_avg,rank_nnz_max,rank_nnz_max_over_avg,"
              "total_time_min,total_time_avg,total_time_max,total_time_max_"
              "over_avg,"
              "mpi_file_read_time_min,mpi_file_read_time_avg,mpi_file_read_"
              "time_max,mpi_file_read_time_max_over_avg,"
              "alltoallv_entries_time_min,alltoallv_entries_time_avg,alltoallv_"
              "entries_time_max,alltoallv_entries_time_max_over_avg,"
              "entries_received_min,entries_received_avg,entries_received_max,"
              "entries_received_max_over_avg,"
              "entry_exchange_bytes_received_min,entry_exchange_bytes_received_"
              "avg,entry_exchange_bytes_received_max,entry_exchange_bytes_"
              "received_max_over_avg\n");
    }

    fprintf(parsing_file, "%s,%d,%s", run_id, comm_size, matrix);

    parsing.rank_nnz.write_csv(parsing_file);
    parsing.total_time.write_csv(parsing_file);
    parsing.mpi_file_read_time.write_csv(parsing_file);
    parsing.alltoallv_entries_time.write_csv(parsing_file);
    parsing.entries_received.write_csv(parsing_file);
    parsing.entry_exchange_bytes_received.write_csv(parsing_file);

    fprintf(parsing_file, "\n");
    fflush(parsing_file);
    fclose(parsing_file);

    bool compute_exists = access(compute_name, F_OK) == 0;
    FILE *compute_file = fopen(compute_name, compute_exists ? "a" : "w");

    if (compute_file == nullptr) {
      return 2;
    }

    if (!compute_exists) {
      fprintf(compute_file,
              "run_id,nprocs,matrix,result_type,"
              "total_flops,error,"
              "rank_flops_min,rank_flops_avg,rank_flops_max,rank_flops_max_"
              "over_avg,"
              "global_kernel_gflops,"
              "global_kernel_plus_full_ghost_gflops,"
              "global_kernel_plus_alltoallv_ghost_gflops,"
              "ghost_exchange_time_min,ghost_exchange_time_avg,ghost_exchange_"
              "time_max,ghost_exchange_time_max_over_avg,"
              "ghost_exchange_alltoallv_time_min,ghost_exchange_alltoallv_time_"
              "avg,ghost_exchange_alltoallv_time_max,ghost_exchange_alltoallv_"
              "time_max_over_avg,"
              "ghost_exchange_time_mean_min,ghost_exchange_time_mean_avg,ghost_"
              "exchange_time_mean_max,ghost_exchange_time_mean_max_over_avg,"
              "ghost_exchange_alltoallv_time_mean_min,ghost_exchange_alltoallv_"
              "time_mean_avg,ghost_exchange_alltoallv_time_mean_max,ghost_"
              "exchange_alltoallv_time_mean_max_over_avg,"
              "ghost_values_received_min,ghost_values_received_avg,ghost_"
              "values_received_max,ghost_values_received_max_over_avg,"
              "ghost_bytes_sent_min,ghost_bytes_sent_avg,ghost_bytes_sent_max,"
              "ghost_bytes_sent_max_over_avg,"
              "ghost_bytes_received_min,ghost_bytes_received_avg,ghost_bytes_"
              "received_max,ghost_bytes_received_max_over_avg,"
              "kernel_time_min,kernel_time_avg,kernel_time_max,kernel_time_max_"
              "over_avg,"
              "kernel_time_mean_min,kernel_time_mean_avg,kernel_time_mean_max,"
              "kernel_time_mean_max_over_avg\n");
    }

    for (size_t i = 0; i < sizeof(results) / sizeof(results[0]); i++) {
      const mpi_compute_benchmark_results *r = &results[i];

      if (r->result_type[0] == '\0') {
        continue;
      }

      fprintf(compute_file, "%s,%d,%s,%s,%llu,%.10f", run_id, comm_size, matrix,
              r->result_type, static_cast<unsigned long long>(r->total_flops),
              r->error);

      r->rank_flops.write_csv(compute_file);

      fprintf(compute_file, ",%.10f,%.10f,%.10f", r->global_kernel_gflops,
              r->global_kernel_plus_full_ghost_gflops,
              r->global_kernel_plus_alltoallv_ghost_gflops);

      r->ghost_exchange_time.write_csv(compute_file);
      r->ghost_exchange_alltoallv_time.write_csv(compute_file);
      r->ghost_exchange_time_mean.write_csv(compute_file);
      r->ghost_exchange_alltoallv_time_mean.write_csv(compute_file);

      r->ghost_values_received.write_csv(compute_file);
      r->ghost_bytes_sent.write_csv(compute_file);
      r->ghost_bytes_received.write_csv(compute_file);

      r->kernel_time.write_csv(compute_file);
      r->kernel_time_mean.write_csv(compute_file);

      fprintf(compute_file, "\n");
    }

    fflush(compute_file);
    fclose(compute_file);

    return 0;
  }
};

void benchmark_exchange_columns(
    const idxtype *global_acol, idxtype *editable_acol, idxtype local_nnz,
    idxtype total_dense_vec_size, const vtype *local_dense_vec,
    int local_dense_count, int rank, int comm_size,
    vtype **out_extended_dense_vec, int *out_extended_dense_vec_count,
    int *out_received_columns_count, mpi_compute_benchmark_results *results);

void benchmark_exchange_columns_gpu(
    const idxtype *d_global_acol, idxtype *d_editable_acol, idxtype local_nnz,
    idxtype total_dense_vec_size, const vtype *d_local_dense_vec,
    int local_dense_count, int rank, int comm_size,
    vtype **out_d_extended_dense_vec, int *out_extended_dense_vec_count,
    int *out_received_columns_count, mpi_compute_benchmark_results *results);

#endif
