# CMAN Arithmetic Intensity Analysis

## Task 1 

Kernel: SpatialTiledReservoir.get_feedback_map, Layer 0.
Source file: tiled14.py, class ImprovedRCDFATrainer.
Profiling evidence: 76.3% of total training step time, from cProfile over 3 batches in codefest/cf02/profiling/project_profile.txt.

The inner loop is s = tanh(s @ W_res.T) repeated T times over all SP = B x H x W spatial locations. This is a recurrent echo-state network applied in parallel over the full spatial extent of the encoder error map.

Reuse pattern: This kernel does not fit the standard GEMM weight-reuse pattern. The recurrent weights W_res and W_lat (128 x 128 = 65 KB each) are tiny and remain stationary across all T steps. The large operand is the state tensor s, which has shape SP x N = 262,144 x 128 = 134 MB and is read and written on every recurrent step. The applicable pattern is streaming recurrent state with stationary weights: weights fit in L1 cache and are perfectly reused; state tensor dominates bandwidth.

Kernel dimensions and data type:

| Symbol | Value | Source |
|--------|-------|--------|
| B | 64 | BATCH_SIZE in tiled14.py |
| C | 64 | encoder layer 0 output channels (ENC_SPECS[0]) |
| H, W | 64, 64 | encoder layer 0 spatial dims (ENC_SPECS[0]) |
| N = 2 x C | 128 | reservoir nodes |
| T | 20 | recurrent steps (PER_LAYER_T[0]) |
| SP = B x H x W | 262,144 | total spatial locations |
| Data type | FP32 | 4 bytes per element |

## Task 2 

One invocation of Layer 0 get_feedback_map at the design operating point.

Step 1: Input normalisation (L2 norm plus divide per spatial location):

    FLOPS_NORM = SP x (2C + C) = 262,144 x 192 = 50,331,648

Step 2: Input projection s = tanh(e_flat @ W_in.T), matrix multiply (SP, C) x (C, N):

    FLOPS_WIN = 2 x SP x C x N = 2 x 262,144 x 64 x 128 = 4,294,967,296

Step 3: T-1 = 19 recurrent steps. Each step runs s @ W_res.T and s_nb @ W_lat.T, both (SP, N) x (N, N):

    FLOPS_PER_STEP     = 2 x SP x N x N = 2 x 262,144 x 128 x 128 = 8,589,934,592
    FLOPS_LATERAL_STEP = 2 x SP x N x N = 8,589,934,592
    FLOPS_RECUR        = 19 x (8,589,934,592 + 8,589,934,592) = 326,417,514,496

Step 4: Readout index gather (negligible):

    FLOPS_READ = SP x C = 262,144 x 64 = 16,777,216

Total:

    TOTAL_FLOPS = 50,331,648 + 4,294,967,296 + 326,417,514,496 + 16,777,216
                = 330,779,590,656 FLOPs  (330.8 GFLOP)

The recurrent steps account for 98.7% of all FLOPs.

## Task 3

No-reuse lower bound: every operand fetched from off-chip DRAM on each access. All FP32 = 4 bytes.

| Operand | Shape | Bytes |
|---------|-------|-------|
| Input error map e_flat | SP x C | 262,144 x 64 x 4 = 67,108,864 |
| Input weights W_in | N x C | 128 x 64 x 4 = 32,768 |
| Recurrent weights W_res | N x N | 128 x 128 x 4 = 65,536 |
| Lateral weights W_lat | N x N | 128 x 128 x 4 = 65,536 |
| State s (19 reads + 19 writes) | 19 x 2 x SP x N | 19 x 2 x 262,144 x 128 x 4 = 5,100,273,664 |
| Output feedback map | SP x C | 262,144 x 64 x 4 = 67,108,864 |
| Total | | 5,234,655,232 bytes (5.23 GB) |

Note: CF02 reported the state term as 2,550,136,832 bytes. The formula 19 x 2 x 262,144 x 128 x 4 evaluates to 5,100,273,664 bytes. CF02 had a transcription error of 2x in the reported result. The corrected total is 5.23 GB.

Full-reuse upper bound: W_in, W_res, W_lat remain on-chip across all T steps. State s evolves in-place without off-chip traffic. This corresponds to the physical analog hardware where weights are device capacitances and state evolves within the network.

| Operand | Bytes |
|---------|-------|
| Input error map e_flat | 67,108,864 |
| Output feedback map | 67,108,864 |
| Total | 134,217,728 bytes (128 MB) |

## Task 4 

    AI_lower (no reuse)   = 330,779,590,656 / 5,234,655,232  = 63.19 FLOP/byte
    AI_upper (full reuse) = 330,779,590,656 / 134,217,728     = 2,464.5 FLOP/byte

Target hardware: 128x128 memcapacitive crossbar (1 GHz) plus sky130A digital FSM (100 MHz, post-route M3 STA nom_tt_025C_1v80).

| Parameter | Hardware | CPU Baseline |
|-----------|----------|--------------|
| Peak compute | 32,768 GFLOP/s (128x128 x 2 x 1 GHz analog) | 1,382 GFLOP/s (AVX2 FP32) |
| Peak bandwidth | 256 GB/s (on-chip SRAM) | 102 GB/s (DDR5-6400 dual) |
| Ridge point | 128 FLOP/byte | 13.5 FLOP/byte |

Roofline positions:

- AI_lower = 63.19 on CPU: above ridge 13.5, compute-bound, attainable = 1,382 GFLOP/s.
- AI_lower = 63.19 on HW: below ridge 128, memory-bound, attainable = 256 GB/s x 63.19 = 16,177 GFLOP/s.
- AI_upper = 2,464.5 on HW: above ridge 128, compute-bound, attainable = 32,768 GFLOP/s (ceiling).

Roofline sketch: codefest/cf09/cman_roofline_sketch.png.

## Task 5 
The simulation-measured throughput is 2,294 GFLOP/s per Layer 0 call (55 cycles x 10 ns = 550 ns per call at 100 MHz, 1.262 MFLOP per call). This is 7x below the memory-bound attainable of 16,177 GFLOP/s and far below the analog ceiling.

The design is limited by the interface bandwidth in the serialization sense: the AXI-S interface processes one spatial location (64 channels) per call. This forces 262,144 sequential calls per Layer 0 training step rather than one. The analog crossbar bandwidth and compute capacity are not the bottleneck. The bottleneck is the digital interface width.

The single highest-leverage change is to widen the AXI-S input transaction to accept a full spatial tile rather than one spatial location. Widening from 64 channels to 4,096 spatial locations x 64 channels per call reduces the calls-per-step from 262,144 to 64 and multiplies throughput by the same factor, raising the projected end-to-end speedup from 3.42x to well above 50x.
