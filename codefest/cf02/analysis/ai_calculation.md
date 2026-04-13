# Arithmetic Intensity — RC-DFA Reservoir Feedback Kernel

## ECE 510 Spring 2026 — Codefest 2 CLLM

## Dominant Kernel Identification

The dominant kernel is **`SpatialTiledReservoir.get\_feedback\_map` (Layer 0)**,
accounting for **76.3% of total step runtime** (6.091s out of 11.095s total across
3 profiled batches, from `project\_profile.txt`). The inner recurrent loop —
`s = tanh(s @ W\_res.T)` repeated T=20 times over B×H×W = 64×64×64 = 262,144
spatial locations — is the bottleneck by cumulative time. This loop simulates
echo-state reservoir dynamics: a fixed random recurrent network producing a
nonlinear projection of the error signal into a high-dimensional state space.

\---

## FLOPs Derivation — Layer 0 Reservoir

**Hyperparameters (from source, `ImprovedRCDFATrainer`):**

|Symbol|Value|Source|
|-|-|-|
|B|64|BATCH\_SIZE in main|
|C|64|encoder layer 0 output channels|
|H, W|64, 64|encoder layer 0 output spatial dims|
|N = 2×C|128|reservoir nodes (SpatialTiledReservoir)|
|T|20|recurrent steps (PER\_LAYER\_T\[0])|
|SP = B×H×W|262,144|total spatial locations|

**Step 1 — Input normalisation**

Each of SP spatial locations has a C-dim vector. L2 norm: 2C FLOPs, divide: C FLOPs.

&#x20;   FLOPS\_NORM = SP × (2C + C) = 262,144 × 192 = 50,331,648


**Step 2 — Input projection: `s = tanh( e\_flat @ W\_in.T )`**

Matrix multiply: (SP, C) × (C, N) → (SP, N).

&#x20;   FLOPS\_WIN = 2 × SP × C × N = 2 × 262,144 × 64 × 128 = 4,294,967,296


**Step 3 — T−1 = 19 recurrent steps: `s = tanh( s @ W\_res.T )`**

Each step: (SP, N) × (N, N) → (SP, N), cost 2×SP×N².
Lateral coupling (`s\_nb @ W\_lat.T`) adds the same cost per step.

&#x20;   FLOPS\_PER\_STEP     = 2 × SP × N² = 2 × 262,144 × 16,384 = 8,589,934,592
    FLOPS\_LATERAL\_STEP = 2 × SP × N² = 8,589,934,592
    FLOPS\_RECUR = 19 × (8,589,934,592 + 8,589,934,592) = 326,417,514,496


**Step 4 — Readout (index gather, negligible)**

&#x20;   FLOPS\_READ ≈ SP × C = 262,144 × 64 = 16,777,216


**Total FLOPs:**

&#x20;   TOTAL\_FLOPS = 50,331,648 + 4,294,967,296 + 326,417,514,496 + 16,777,216
               = 330,779,590,656  ≈ 330.8 GFLOP


\---

## Bytes Transferred (DRAM, no reuse — FP32 = 4 bytes)

|Operand|Shape|Bytes|
|-|-|-|
|Error map e\_flat|(262,144 × 64)|67,108,864|
|Input weights W\_in|(128 × 64)|32,768|
|Recurrent weights W\_res|(128 × 128)|65,536|
|Lateral weights W\_lat|(128 × 128)|65,536|
|State s read+write over T−1 steps|19 × 2 × (262,144 × 128) × 4|2,550,136,832|
|Output fb|(262,144 × 64)|67,108,864|
|**Total**||**2,684,518,400 bytes ≈ 2.68 GB**|

The state tensor s dominates bytes transferred, accounting for 95% of total DRAM
traffic. W\_res and W\_lat are only 65 KB each and would fit in L1 cache on any
hardware — the memory pressure comes entirely from the 128 MB state tensor being
read and written 19 times.

\---

## Arithmetic Intensity

&#x20;   AI = TOTAL\_FLOPS / TOTAL\_BYTES
       = 330,779,590,656 / 2,684,518,400
       = 123.2 FLOP/byte


**The dominant kernel has an arithmetic intensity of 123.2 FLOP/byte.**

\---

## Roofline Position — CPU (Software Baseline)

Target hardware: Intel Core Ultra 9 275HX (source: Intel ARK + cpu-monkey.com)

* Peak FP32 (AVX2, 8 P-cores × 5.4 GHz × 16 FP32/cycle): **1,382 GFLOP/s**
* Peak DRAM bandwidth (DDR5-6400, dual channel): **102 GB/s**
* Ridge point: 1,382 / 102 = **13.5 FLOP/byte**

AI = 123.2 >> 13.5 → kernel is **compute-bound** on CPU.

Observed throughput: 6.091s for 12 reservoir calls → 0.508s per call.

&#x20;   Observed GFLOP/s = 330.8 GFLOP / 0.508s = 651 GFLOP/s
    CPU utilisation  = 651 / 1,382 = 47%


\---

## Roofline Position — Memcapacitive Hardware Design Point

The proposed accelerator implements the reservoir dynamics as a **physical analog
process** rather than computing them digitally. W\_res and W\_lat are encoded as
device capacitances at fabrication — they have zero memory bandwidth cost because
they are physical properties of the network, not data. The state s evolves
in-place within the network without any memory read/write. The only DRAM traffic
is the input error map and the output feedback map.

Effective bytes with physical reservoir (input + output only):

&#x20;   BYTES\_HW = (SP×C + SP×C) × 4 = 2 × 262,144 × 64 × 4 = 134,217,728 bytes

    HW\_AI = 330,779,590,656 / 134,217,728 = 2,464 FLOP/byte


Hardware specs (128×128 memcapacitive network, 1 GHz sampling rate):

* Peak equivalent throughput: 128 × 128 × 2 × 1 GHz = **32,768 GFLOP/s**
* On-chip SRAM bandwidth (input/output staging): **256 GB/s**
* Ridge point: 32,768 / 256 = **128 FLOP/byte**

HW\_AI = 2,464 >> HW ridge 128 → kernel is **compute-bound on the accelerator**.
Attainable = min(256 × 2,464, 32,768) = **32,768 GFLOP/s** (hits compute ceiling).

Expected kernel speedup: 32,768 / 651 = **50×**
Amdahl end-to-end (kernel = 76.3%): 1 / (0.237 + 0.763/50) = **4.0×**

