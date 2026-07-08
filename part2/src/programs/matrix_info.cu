#include "parse_file.cuh"

#include <math.h>
#include <stdio.h>
#include <unistd.h>

int write_info(const char *name, const char *mat_name, idxtype rows,
               idxtype cols, idxtype nnz, bool symmetric, bool graph,
               double max_to_avg, double coeff_variation) {
  bool exists = false;

  if (access(name, F_OK) == 0) {
    exists = true;
  }

  FILE *f = fopen(name, exists ? "a" : "w");

  if (f == NULL) {
    return 1;
  }

  if (!exists) {
    fprintf(f, "%60s,%15s,%15s,%15s,%15s,%15s,%15s,%15s,%15s,%15s\n",
            "matrix_name", "sparsity", "max to avg", "coeff variation", "rows",
            "cols", "nnz", "symmetric", "graph", "total elements");
  }

  double sparsity = (double)1.0 - ((double)nnz / ((double)rows * (double)cols));
  fprintf(f, "%60s,%15f,%15f,%15f,%15ld,%15ld,%15ld,%15s,%15s,%15ld\n",
          mat_name, sparsity, max_to_avg, coeff_variation, rows, cols, nnz,
          symmetric ? "true" : "false", graph ? "true" : "false", rows * cols);
  fflush(f);

  fclose(f);

  return 0;
}

void compute_regularity(idxtype *csr_arow, idxtype rows, idxtype nnz,
                        double *max_to_avg_ratio_out,
                        double *coeff_variation_out) {
  double avg_num_of_elements = (double)nnz / rows;
  idxtype max_elements = 0;
  double standard_deviation = 0;

  for (idxtype i = 0; i < rows; i++) {
    idxtype elements = csr_arow[i + 1] - csr_arow[i];

    if (max_elements < elements) {
      max_elements = elements;
    }

    standard_deviation += pow(((double)elements - avg_num_of_elements), 2);
  }

  standard_deviation /= rows;
  standard_deviation = sqrt(standard_deviation);

  double max_to_avg_ratio = (double)max_elements / avg_num_of_elements;
  double coeff_variation = standard_deviation / avg_num_of_elements;

  *max_to_avg_ratio_out = max_to_avg_ratio;
  *coeff_variation_out = coeff_variation;
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <matrix file>\n", argv[0]);
    return 1;
  }

  printf("%s: working on %s\n", __FILE__, argv[1]);

  idxtype *host_acol, *host_csr_arow;
  vtype *host_aval;
  idxtype rows, cols, nnz;
  bool graph, symmetric;
  if (parse_file_csr(argv[1], &host_csr_arow, &host_acol, &host_aval, &rows, &cols, &nnz,
                     &graph, &symmetric) != 0) {
    printf("error encountered when parsing file\n");
    return 2;
  }

  double max_to_avg, coeff_variation;
  compute_regularity(host_csr_arow, rows, nnz, &max_to_avg, &coeff_variation);

  if (write_info("matrix_info.csv", argv[1], rows, cols, nnz, symmetric, graph,
                 max_to_avg, coeff_variation) != 0) {
    printf("could not write results to file\n");
    return 3;
  }

  free(host_csr_arow);
  free(host_acol);
  free(host_aval);

  printf("done\n");

  return 0;
}
