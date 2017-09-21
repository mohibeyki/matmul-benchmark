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

void print_matrix_2D(double **A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            printf("%f ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

void matrix_mult(double **a, double **b, double **c, int m, int n, int k)
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
    if (argc < 4)
    {
        printf("use ./gdmm m n k\n");
        return 1;
    }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);

    int *row_min_array = new int[m];
    double **a = new double *[m];
    double **b = new double *[n];
    double **c = new double *[m];

    for (int i = 0; i < m; ++i)
        a[i] = new double[n];

    for (int i = 0; i < n; ++i)
        b[i] = new double[k];

    for (int i = 0; i < m; ++i)
        c[i] = new double[k];

    srand(time(0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = i * j % 1024 + 1;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            b[i][j] = i * j % 1024 + 1;

    double start_matrix_mullt = get_time();
    matrix_mult(a, b, c, m, n, k);
    printf("%f\n", get_time() - start_matrix_mullt);
    return 0;
}
