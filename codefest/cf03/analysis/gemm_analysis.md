# GEMM Kernel Analysis — RTX 5070 Ti (sm_120, Blackwell)

## (a) Why the Naive Kernel is Memory-Bound

The naive kernel assigns one thread per output element `C[i][j]`, which
computes a dot product over the full K dimension. Each thread independently
reads an entire row of A and an entire column of B from global memory, with
no data reuse across threads. For N=1024 this produces roughly 8.6 GB of
global memory traffic per kernel launch (2·N³ floats read, N² written),
yielding an arithmetic intensity of only ~0.25 FLOP/byte against DRAM —
far below the RTX 5070 Ti's ridge point of ~101 FLOP/byte. In practice,
Blackwell's large L1 cache absorbs many of those reads, pushing the
observed bottleneck into the SM at 95% utilization, but the kernel remains
architecturally memory-bound by design.

## (b) How Tiling Reduces DRAM Traffic

Shared-memory tiling loads a TILE×TILE sub-block of A and B into on-chip
`__shared__` memory once, then reuses it for all TILE output elements in
that tile. Each value fetched from global memory is used TILE times instead
of once, reducing DRAM reads by a factor of TILE. With TILE=8, algorithmic
DRAM traffic drops from ~8.6 GB to ~1.1 GB — an 8× reduction — and
arithmetic intensity rises proportionally to ~2.0 FLOP/byte.

## (c) Did the Tiled Kernel Achieve the Expected Improvement?

No — the improvement was marginal. Achieved performance was 86.2 TFLOP/s
(naive, 95.2% SM util) versus 76.3 TFLOP/s (tiled, 84.3% SM util); the
tiled kernel was actually slightly *slower*. The remaining bottleneck is
**tile size**: TILE=8 means each thread block holds only 64 threads (8×8),
which is too few to hide the latency of two `__syncthreads()` barriers per
tile iteration and too small to amortize shared-memory load overhead. A
standard TILE=32 (1024 threads/block) would increase register reuse,
improve occupancy, and expose the full 32× DRAM traffic reduction needed
to move both kernels off the compute ceiling.
