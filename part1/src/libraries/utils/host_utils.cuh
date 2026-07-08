#ifndef UTILS_H
#define UTILS_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "constants.h"

#define TIMER_DEF(n) struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone *)0)
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone *)0)
#define TIMER_ELAPSED(n)                                                       \
  ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 +                            \
   (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

#define zero_check(value)                                                      \
  {                                                                            \
    auto error = value;                                                        \
    if (error != 0) {                                                          \
      printf("error: %s %d %d\n", __FILE__, __LINE__, error);                  \
      exit(1);                                                                 \
    }                                                                          \
  }

#define bench_expression(expression)                                           \
  {                                                                            \
    TIMER_DEF(timer);                                                          \
    TIMER_START(timer);                                                        \
    expression;                                                                \
    TIMER_STOP(timer);                                                         \
    printf("time passed: %10f\n", TIMER_ELAPSED(timer) / 1e6);                 \
  }

double arithmetic_mean(double *v, int len);
double geometric_mean(double *v, int len);
double sigma_fn_sol(double *v, double mu, int len);

double compute_mse(const valuetype *vec_one, const valuetype *vec_two,
                   const size_t entries);

double compute_deviation(const double *time, const double avg,
                         const size_t elements);

valuetype *init_dense_vec_host(int64_t cols);

#endif
