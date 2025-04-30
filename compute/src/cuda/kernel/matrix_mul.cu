#include <cstdint>
#include <float.h>
#include <stdio.h>

# define TILE_SIZE 16

__global__ void matrix_multiply_shared(
    const double *A, const double *B, double *C, int M, int N, int K)
{
    __shared__ double Asub[TILE_SIZE][TILE_SIZE];
    __shared__ double Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            Asub[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}


extern "C" void launch_matrix_multiply(const double *hA, const double *hB, double *hC, int M, int N, int K)
{
    // 1. Size bookkeeping
    size_t sizeA = static_cast<size_t>(M) * N * sizeof(double);
    size_t sizeB = static_cast<size_t>(N) * K * sizeof(double);
    size_t sizeC = static_cast<size_t>(M) * K * sizeof(double);

    // 2. Device buffers
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    // 3. Copy inputs
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // 4. Launch
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matrix_multiply_shared<<<grid, block>>>(dA, dB, dC, M, N, K);

    // 5. Error checks
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));

    // 6. Copy result back
    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // 7. Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
