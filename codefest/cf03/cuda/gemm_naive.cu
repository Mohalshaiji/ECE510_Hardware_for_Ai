// gemm_naive.cu
// Naive GEMM: one thread per output element C[i][j] = sum_k A[i][k] * B[k][j]

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  // Matrix size N x N

// ─────────────────────────────────────────────
// Kernel: one thread computes one output element
// ─────────────────────────────────────────────
__global__ void gemm_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
void fill_random(float* M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (float)rand() / RAND_MAX;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int n = N;
    size_t bytes = (size_t)n * n * sizeof(float);

    // Host allocation
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    fill_random(hA, n);
    fill_random(hB, n);

    // Device allocation
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // Grid / block dimensions
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    gemm_naive<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    gemm_naive<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 2 * N^3 FLOPs for GEMM
    double flops   = 2.0 * (double)n * n * n;
    double gflops  = flops / (ms * 1e-3) / 1e9;

    printf("=== Naive GEMM (N=%d) ===\n", n);
    printf("  Elapsed time : %.3f ms\n", ms);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
