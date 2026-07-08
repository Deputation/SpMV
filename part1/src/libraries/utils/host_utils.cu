#include "host_utils.cuh"

double arithmetic_mean(double *v, int len) {

  double mu = 0.0;
  for (int i = 0; i < len; i++)
    mu += (double)v[i];
  mu /= (double)len;

  return (mu);
}

double geometric_mean(double *v, int len) {
  double mu = 1.0;
  for (int i = 0; i < len; i++) {
    mu *= (v[i] > 0) ? ((double)v[i]) : 1;
  }
  mu = pow(mu, 1.0 / len);

  return (mu);
}

double sigma_fn_sol(double *v, double mu, int len) {
  double sigma = 0.0;
  for (int i = 0; i < len; i++) {
    sigma += ((double)v[i] - mu) * ((double)v[i] - mu);
  }
  sigma /= (double)len;

  return (sigma);
}

double compute_mse(const valuetype *vec_one, const valuetype *vec_two,
                   const size_t entries) {
  double *error = (double *)malloc(sizeof(double) * entries);
  assert(error != nullptr);
  memset(error, 0, sizeof(double) * entries);

  for (size_t i = 0; i < entries; i++) {
    error[i] = pow(vec_one[i] - vec_two[i], 2);
  }

  double total_error = 0;
  for (size_t i = 0; i < entries; i++) {
    total_error += error[i];
  }
  total_error /= entries;

  free(error);

  return total_error;
}

double compute_deviation(const double *time, const double avg,
                         const size_t elements) {
  double sum = 0;
  for (size_t i = 0; i < elements; i++) {
    sum += pow(time[i] - avg, 2);
  }
  sum /= elements - 1;
  sum = sqrt(sum);
  return sum;
}

valuetype *init_dense_vec_host(int64_t cols) {
  valuetype *dense_vec = (valuetype *)malloc(sizeof(valuetype) * cols);
  assert(dense_vec != nullptr);

  for (int64_t i = 0; i < cols; i++) {
    dense_vec[i] = (rand() % 100) / 100.f;
  }

  return dense_vec;
}
