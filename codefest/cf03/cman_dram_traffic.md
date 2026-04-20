Cman Codefest 3

1) 
Each output element C[i][j] needs N reads from A and N from B, so 2N accesses per output. Each element of B gets read N=32 times or once for every output row. Over all N^2 outputs, total accesses = N^2 * 2N = 2N^3 = 65,536 elements, so DRAM traffic = 65,536 * 4 = 262,144 bytes.

2) 
There are (N/T)^2 = 16 output tiles, each stepping through N/T = 4 tile pairs along K. Every step loads a T×T block of A and one of B. Total elements loaded = 2 * 16 * 4 * 64 = 8,192 (which is just 2N^3/T), so DRAM traffic = 8,192 * 4 = 32,768 bytes.

3)
262,144 / 32,768 = 8 = T. The ratio equals T because each element brought into shared memory gets reused T times across the tile, so you replace T DRAM reads with one.

4)
FLOPs for both = 2N^3 = 65,536, giving t_compute = 65,536 / 10e12 = 6.6 ns.
Naive: t_mem = 262,144 / 320e9 = 819 ns, so execution time is 819 ns and the bottleneck is memory.
Tiled: t_mem = 32,768 / 320e9 = 102 ns, so execution time is 102 ns and still memory-bound.

Tiling gives an 8x speedup but both cases stay memory-bound since compute finishes roughly 100x faster than memory can keep up. N=32 is just too small for this hardware to flip to compute-bound.
