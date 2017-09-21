#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#define BLOCK_SIZE 16

#ifndef DEVICE_COUNT
#define DEVICE_COUNT 1
#endif

void print_matrix_2D(double *A, int rows, int cols)
{
        for (int i = 0; i < rows; ++i)
        {
                for (int j = 0; j < cols; ++j)
                        printf("%.0f ", A[i * cols + j]);
                printf("\n");
        }
        printf("\n");
}

__global__ void gpu_matrix_mult(double *a, double *b, double *c, int ms, int me, int n, int k)
{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0;
        if (col < k && row >= ms && row < me)
        {
                for (int i = 0; i < n; i++)
                        sum += a[row * n + i] * b[i * k + col];
                c[row * k + col] = sum;
        }
}

int main(int argc, char *argv[])
{
        if (argc < 4)
        {
                printf("use ./gdmm_cuda m n k\n");
                return 1;
        }
        int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);

        double *a = new double[m * n];
        double *b = new double[n * k];
        double *c = new double[m * k];
        int *min_rows = new int[m];

        srand(time(0));
        for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                        a[i * n + j] = rand() % 1024 + 1;

        for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                        b[i * k + j] = rand() % 1024 + 1;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double *d_a[DEVICE_COUNT], *d_b[DEVICE_COUNT], *d_c[DEVICE_COUNT];
        int *d_min_rows[DEVICE_COUNT];

        for (int i = 0; i < DEVICE_COUNT; i++)
        {
                cudaSetDevice(i);
                cudaMalloc((void **)&d_a[i], sizeof(double) * m * n);
                cudaMalloc((void **)&d_b[i], sizeof(double) * n * k);
                cudaMalloc((void **)&d_c[i], sizeof(double) * m * k);
                cudaMalloc((void **)&d_min_rows[i], sizeof(int) * m);
                cudaMemcpy(d_a[i], a, sizeof(double) * m * n, cudaMemcpyHostToDevice);
                cudaMemcpy(d_b[i], b, sizeof(double) * n * k, cudaMemcpyHostToDevice);
                cudaMemcpy(d_min_rows[i], min_rows, sizeof(int) * m, cudaMemcpyHostToDevice);
        }

        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 numBlocks(grid_cols, grid_rows);
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

        cudaEventRecord(start, 0);
        for (int i = 0; i < DEVICE_COUNT; i++)
        {
                cudaSetDevice(i);
                int ms = m / DEVICE_COUNT * i, me = m / DEVICE_COUNT * (i + 1);
                int device_mem = m * k / DEVICE_COUNT;
                gpu_matrix_mult<<<numBlocks, blockSize>>>(d_a[i], d_b[i], d_c[i], ms, me, n, k);
                cudaMemcpy(c + (device_mem * i), d_c[i] + (device_mem * i), sizeof(double) * device_mem, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
        }
        float gpu_elapsed_time_ms;
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("%f\n", gpu_elapsed_time_ms);

        return 0;
}
