// gemm_tiled.cu
// Shared-memory tiled GEMM — tile size 8
// Each thread block loads a TILE x TILE sub-matrix of A and B into shared
// memory, computes a partial dot product, then advances to the next tile.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N    1024   // Matrix size N x N
#define TILE 8      // Tile size (shared-memory tile)

// ─────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────
__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*                    C,
                            int                       n) {
    // Shared memory tiles for A and B
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles needed to span the K dimension
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // ── Load tile of A (row, t*TILE + tx) ──
        int aCol = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;

        // ── Load tile of B (t*TILE + ty, col) ──
        int bRow = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        __syncthreads();  // wait until both tiles are loaded

        // ── Compute partial dot product for this tile ──
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();  // wait before overwriting shared memory
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
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

    // Grid / block dimensions — each block is exactly one TILE x TILE
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 2 * N^3 FLOPs for GEMM
    double flops  = 2.0 * (double)n * n * n;
    double gflops = flops / (ms * 1e-3) / 1e9;

    printf("=== Tiled GEMM (N=%d, TILE=%d) ===\n", n, TILE);
    printf("  Elapsed time : %.3f ms\n", ms);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
