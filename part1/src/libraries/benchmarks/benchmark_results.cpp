#include "benchmark_results.h"

#include <cstdlib>
#include <stdio.h>
#include <unistd.h>

void string_conversion(char *output, size_t output_size, double value) {
  if (value == -1.f) {
    strcpy(output, "N/A");
    return;
  }

  if (snprintf(output, output_size, "%20f", value) == 0) {
    printf("error: %s %d\n", __FILE__, __LINE__);
    exit(1);
  }
}

int write_results(const char *name, const char *mat_name,
                  const benchmark_results results[], const size_t elements) {
  bool exists = false;

  if (access(name, F_OK) == 0) {
    exists = true;
  }

  FILE *f = fopen(name, exists ? "a" : "w");

  if (f == NULL) {
    return 1;
  }

  if (!exists) {
    fprintf(f, "%60s,%20s,%20s,%20s,%20s,%20s,%20s,%20s,%25s,%25s,%20s\n",
            "matrix_name", "method", "math_time_ari (s)", "math_time_geo (s)",
            "math_time_dev (s)", "pre_time_ari (s)", "pre_time_geo (s)",
            "pre_time_dev (s)", "gflops (GFLOPS/s)", "eff_bw (GB/s)", "mse");
  }

  for (size_t i = 0; i < elements; i++) {
    // only those that might equal -1.f
    char time_avg_str[32], time_avg_geo[32], time_dev[32], pre_time_avg[32],
        pre_time_geo[32], pre_time_dev[32], eff_bw[32];

    string_conversion(time_avg_str, 32, results[i].time_average);
    string_conversion(time_avg_geo, 32, results[i].time_average_geo);
    string_conversion(time_dev, 32, results[i].time_deviation);

    string_conversion(pre_time_avg, 32, results[i].preprocessing_time_average);
    string_conversion(pre_time_geo, 32,
                      results[i].preprocessing_time_average_geo);
    string_conversion(pre_time_dev, 32,
                      results[i].preprocessing_time_deviation);

    string_conversion(eff_bw, 32, results[i].effective_bw);

    fprintf(f, "%60s,%20s,%20s,%20s,%20s,%20s,%20s,%20s,%25f,%25s,%20f\n",
            mat_name, results[i].method, time_avg_str, time_avg_geo, time_dev,
            pre_time_avg, pre_time_geo, pre_time_dev, results[i].gflops, eff_bw,
            results[i].error);
    fflush(f);
  }

  fclose(f);

  return 0;
}
