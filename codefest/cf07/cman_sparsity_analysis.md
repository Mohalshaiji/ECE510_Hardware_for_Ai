# CMAN Sparsity Breakeven Analysis

N = 512, s = fraction of zeros, nnz = (1-s)N^2

## Task 1

**(a) Dense MVM compute**

2N^2 = 524,288 FLOPs

**(b) Dense memory**

4N^2 = 1,048,576 bytes

**(c) Sparse compute**

Only nonzero elements produce MACs:

2(1-s)N^2 = 524,288(1-s) FLOPs

**(d) Sparse memory (CSR)**

CSR stores a values array and a column index array (each of length nnz, 4 bytes each) and a row pointer array of length N+1 (4 bytes each):

8N^2(1-s) + 4(N+1) = 2,097,152(1-s) + 2,052 bytes

## Task 2

Speedup = Dense FLOPs / Sparse FLOPs = 2N^2 / 2(1-s)N^2 = 1/(1-s)

Setting speedup = 2: 1/(1-s) = 2, so **s = 0.5**

## Task 3

Set sparse memory equal to dense memory and solve for s:

8N^2(1-s) + 4(N+1) = 4N^2

8N^2 - 8N^2*s + 4N + 4 = 4N^2

8N^2*s = 4N^2 + 4N + 4

s = (4N^2 + 4N + 4) / 8N^2 = 1/2 + 1/(2N) + 1/(2N^2)

Plugging in N = 512:

s = 0.5 + 0.000977 + 0.0000019 = **0.5010**

Above ~50.1% sparsity, CSR uses less memory than dense. The breakeven is so close to 0.5 because the row pointer overhead (4(N+1) = 2,052 bytes) is negligible relative to the weight matrix (~1 MB).

## Task 4

For a memory bandwidth limited system, execution time scales directly with bytes transferred.

Dense time = 1,048,576 bytes / 320 GB/s = 3.28 microseconds

Sparse memory at s=0.9 = 2,097,152 * 0.1 + 2,052 = 211,767 bytes

Sparse time = 211,767 bytes / 320 GB/s = 0.662 microseconds

Speedup = 3.28 / 0.662 = **4.95x**
