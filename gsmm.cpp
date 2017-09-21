#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <omp.h>

double get_time()
{
    struct timeval ret;
    gettimeofday(&ret, NULL);
    return ((ret.tv_sec) * 1000000u + ret.tv_usec) / 1.e3;
}

void print_matrix_2D(float **A, int nr_rows_A, int nr_cols_A)
{
    for (int i = 0; i < nr_rows_A; ++i)
    {
        for (int j = 0; j < nr_cols_A; ++j)
            printf("%f ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

void matrix_mult(float **a, float **b, float **c, int m, int n, int k)
{
#pragma omp parallel shared(a, b, c)
    {
        int i, j, l;
#pragma omp for schedule(auto)
        for (i = 0; i < m; i++)
            for (j = 0; j < k; j++)
                for (l = 0; l < n; l++)
                    c[i][j] += a[i][l] * b[l][j];
    }
}

int main(int argc, char *argv[])
{
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);

    int *row_min_array = new int[m];
    float **a = new float *[m];
    float **b = new float *[n];
    float **c = new float *[m];
    for (int i = 0; i < m; ++i)
        a[i] = new float[n];

    for (int i = 0; i < n; ++i)
        b[i] = new float[k];

    for (int i = 0; i < m; ++i)
        c[i] = new float[k];

    srand(time(0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = i * j % 3 + 1;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            b[i][j] = i * j % 3 + 1;

    double start_matrix_mullt = get_time();
    matrix_mult(a, b, c, m, n, k);
    printf("%f\n", get_time() - start_matrix_mullt);

    return 0;
}
