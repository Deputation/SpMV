#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "include/mmio.h"

#define WARMUP 2
#define NITER 10

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


const char *base_name(const char *path) {
    const char *slash = strrchr(path, '/');
    if (slash == NULL) {
        return path;
    }
    return slash + 1;
}


int file_exists_and_nonempty(const char *path) {
    FILE *f = fopen(path, "r");
    long size;

    if (f == NULL) {
        return 0;
    }

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fclose(f);

    return size > 0;
}


void coo_to_csr(int M, int nz, int *I, int *J, double *val,
                int **row_ptr_out, int **col_ind_out, float **values_out) {
    int *row_ptr = (int *)calloc(M + 1, sizeof(int));
    int *col_ind = (int *)malloc(nz * sizeof(int));
    float *values = (float*)malloc(nz * sizeof(float));

    if (row_ptr == NULL || col_ind == NULL || values == NULL) {
        fprintf(stderr, "Allocation failed in coo_to_csr\n");
        exit(1);
    }

    // Step 1: Count number of entries in each row
    for (int i = 0; i < nz; i++) {
        row_ptr[I[i] + 1]++;
    }

    // Step 2: Cumulative sum to get row_ptr
    for (int i = 0; i < M; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Step 3: Fill col_ind and values arrays
    int *temp_row_ptr = (int *)malloc((M + 1) * sizeof(int));
    if (temp_row_ptr == NULL) {
        fprintf(stderr, "Allocation failed for temp_row_ptr\n");
        exit(1);
    }

    for (int i = 0; i <= M; i++) {
        temp_row_ptr[i] = row_ptr[i];
    }

    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = temp_row_ptr[row];

        col_ind[dest] = J[i];
        values[dest] = (float)val[i];

        temp_row_ptr[row]++;
    }

    free(temp_row_ptr);

    *row_ptr_out = row_ptr;
    *col_ind_out = col_ind;
    *values_out = values;
}


// Sparse Matrix-Vector Multiplication: y = A * x
// A is in CSR (Compressed Sparse Row) format
void spmv_csr(
    int rows,
    const int *row_ptr,
    const int *col_ind,
    const float *values,
    const float *x,
    float *y
) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_ind[j]];
        }
    }
}


void write_csv_row(
    const char *matrix_path,
    int M,
    int N,
    int nz_stored,
    int nz_used,
    double read_time_s,
    double convert_time_s,
    double mean_s,
    double std_s,
    double min_s,
    double max_s,
    double gflops,
    double checksum,
    const char *output_dir
) {
    const char *job_id = getenv("SLURM_JOB_ID");
    char csv_path[4096];
    int write_header;
    FILE *csv;

    if (job_id == NULL || strlen(job_id) == 0) {
        job_id = "local";
    }

    snprintf(
        csv_path,
        sizeof(csv_path),
        "%s/baseline_cpu_%s.csv",
        output_dir,
        job_id
    );

    write_header = !file_exists_and_nonempty(csv_path);

    csv = fopen(csv_path, "a");
    if (csv == NULL) {
        fprintf(stderr, "Could not open CSV output file: %s\n", csv_path);
        exit(1);
    }

    if (write_header) {
        fprintf(
            csv,
            "job_id,matrix,rows,cols,nnz_stored,nnz_used,warmup,niter,"
            "read_time_s,convert_time_s,spmv_mean_s,spmv_std_s,"
            "spmv_min_s,spmv_max_s,gflops,checksum\n"
        );
    }

    fprintf(
        csv,
        "%s,%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.9f\n",
        job_id,
        base_name(matrix_path),
        M,
        N,
        nz_stored,
        nz_used,
        WARMUP,
        NITER,
        read_time_s,
        convert_time_s,
        mean_s,
        std_s,
        min_s,
        max_s,
        gflops,
        checksum
    );

    fclose(csv);

    printf("Wrote baseline CSV row to %s\n", csv_path);
}


int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz_stored;
    int nz_used = 0;
    int nz_capacity;
    int i, *I, *J;
    double *val;
    const char *output_dir = ".";

    double read_time_s;
    double convert_time_s;
    double times[NITER];
    double sum = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    double gflops = 0.0;
    double checksum = 0.0;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [output-dir]\n", argv[0]);
        exit(1);
    }

    if (argc >= 3) {
        output_dir = argv[2];
    }

    TIMER_DEF(read_timer);
    TIMER_START(read_timer);

    if ((f = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Could not open input file: %s\n", argv[1]);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
    {
        printf("Sorry, this application only supports sparse Matrix Market matrices.\n");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if (mm_is_complex(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz_stored)) != 0) {
        exit(1);
    }

    /*
     * Preserve the simple professor-style COO read, but allocate room for
     * symmetric expansion when needed so the mathematical matrix matches the
     * Matrix Market header.
     */
    nz_capacity = mm_is_symmetric(matcode) ? 2 * nz_stored : nz_stored;

    I = (int *) malloc(nz_capacity * sizeof(int));
    J = (int *) malloc(nz_capacity * sizeof(int));
    val = (double *) malloc(nz_capacity * sizeof(double));

    if (I == NULL || J == NULL || val == NULL) {
        fprintf(stderr, "Allocation failed while reading matrix\n");
        exit(1);
    }

    for (i = 0; i < nz_stored; i++)
    {
        int row, col;
        double value = 1.0;

        if (mm_is_pattern(matcode)) {
            if (fscanf(f, "%d %d\n", &row, &col) != 2) {
                fprintf(stderr, "Error reading pattern entry %d\n", i);
                exit(1);
            }
        } else {
            if (fscanf(f, "%d %d %lg\n", &row, &col, &value) != 3) {
                fprintf(stderr, "Error reading numeric entry %d\n", i);
                exit(1);
            }
        }

        row--;
        col--;

        I[nz_used] = row;
        J[nz_used] = col;
        val[nz_used] = value;
        nz_used++;

        if (mm_is_symmetric(matcode) && row != col) {
            I[nz_used] = col;
            J[nz_used] = row;
            val[nz_used] = value;
            nz_used++;
        }
    }

    if (f != stdin) fclose(f);

    TIMER_STOP(read_timer);
    read_time_s = TIMER_ELAPSED(read_timer) / 1e6;

#ifdef PRINT_OUTPUT
    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz_stored);
    for (i = 0; i < nz_used; i++) {
        fprintf(stdout, "%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);
    }
#endif

    int *row_ptr, *col_ind;
    float *values, *x, *y;

    x = (float*)malloc(sizeof(float) * N);
    y = (float*)malloc(sizeof(float) * M);

    if (x == NULL || y == NULL) {
        fprintf(stderr, "Allocation failed for dense vectors\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
    }

    TIMER_DEF(convert_timer);
    TIMER_START(convert_timer);

    coo_to_csr(M, nz_used, I, J, val, &row_ptr, &col_ind, &values);

    TIMER_STOP(convert_timer);
    convert_time_s = TIMER_ELAPSED(convert_timer) / 1e6;

    for (int iter = 0; iter < WARMUP; iter++) {
        spmv_csr(M, row_ptr, col_ind, values, x, y);
    }

    for (int iter = 0; iter < NITER; iter++) {
        TIMER_DEF(spmv_timer);
        TIMER_START(spmv_timer);

        spmv_csr(M, row_ptr, col_ind, values, x, y);

        TIMER_STOP(spmv_timer);
        times[iter] = TIMER_ELAPSED(spmv_timer) / 1e6;
    }

    min_time = times[0];
    max_time = times[0];

    for (int iter = 0; iter < NITER; iter++) {
        sum += times[iter];

        if (times[iter] < min_time) {
            min_time = times[iter];
        }

        if (times[iter] > max_time) {
            max_time = times[iter];
        }
    }

    mean = sum / NITER;

    for (int iter = 0; iter < NITER; iter++) {
        double diff = times[iter] - mean;
        variance += diff * diff;
    }

    variance /= NITER;
    stddev = sqrt(variance);

    gflops = (2.0 * (double)nz_used) / (mean * 1e9);

    for (int i = 0; i < M; i++) {
        checksum += y[i];
    }

#ifdef PRINT_OUTPUT
    // Print the result
    printf("Result y = A * x:\n");
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }
#endif

    printf("matrix: %s\n", base_name(argv[1]));
    printf("rows: %d cols: %d nnz_stored: %d nnz_used: %d\n", M, N, nz_stored, nz_used);
    printf("read_time_s: %.9f\n", read_time_s);
    printf("convert_time_s: %.9f\n", convert_time_s);
    printf("spmv_mean_s: %.9f\n", mean);
    printf("spmv_std_s: %.9f\n", stddev);
    printf("spmv_min_s: %.9f\n", min_time);
    printf("spmv_max_s: %.9f\n", max_time);
    printf("gflops: %.6f\n", gflops);
    printf("checksum: %.9f\n", checksum);

    write_csv_row(
        argv[1],
        M,
        N,
        nz_stored,
        nz_used,
        read_time_s,
        convert_time_s,
        mean,
        stddev,
        min_time,
        max_time,
        gflops,
        checksum,
        output_dir
    );

    free(I);
    free(J);
    free(val);
    free(row_ptr);
    free(col_ind);
    free(values);
    free(x);
    free(y);

    return 0;
}
