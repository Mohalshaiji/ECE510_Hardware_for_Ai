# HW/SW Partition Rationale — RC-DFA VAE
## ECE 510 Spring 2026 — Codefest 2 CLLM

## (a) Which Kernel to Accelerate and Why

The `SpatialTiledReservoir.get_feedback_map` kernel is the target for hardware
acceleration across all four encoder layers. From `project_profile.txt`,
`get_feedback_map` accounts for 76.3% of total profiled step time (6.091s tottime
out of 11.095s), with Layer 0 alone consuming 0.508s per call.

The roofline analysis places the kernel at AI = 123.2 FLOP/byte, compute-bound
above the CPU ridge point of 13.5 FLOP/byte. However, the stronger motivation for
hardware acceleration is algorithmic rather than arithmetic: the software
implementation simulates a recurrent dynamical system — `s ← tanh(s @ W_res.T)`
for T=20 steps — by discretising it into sequential digital matrix multiplies.
Each step is causally dependent on the previous, so no step-level parallelism is
possible in software. The reservoir dynamics (fading memory, nonlinear state
evolution, spatial mixing via W_lat) are properties of a continuous physical
system being approximated digitally at unnecessary cost.

A memcapacitive crossbar network is the natural accelerator because it implements
the reservoir dynamics as a physical process rather than computing them. Voltage
inputs drive charge redistribution through the capacitive network; the network
state evolves according to its own physical dynamics; and the T=20 timesteps become
T samples of that evolving state rather than T sequential operations. W_res and
W_lat are encoded as device capacitances at fabrication — they require no memory
traffic, no loading, and no precise numerical representation since reservoir
computing is inherently noise-tolerant. The tanh nonlinearity emerges from device
physics rather than an explicit digital operation.

The proposed design is a 128×128 memcapacitive crossbar network sampling at 1 GHz.
Peak equivalent throughput: 128×128×2×1GHz = 32,768 GFLOP/s. Expected kernel
speedup: 32,768 / 651 = **50×**. Amdahl end-to-end (kernel = 76.3%): **4.0×**.

## (b) What Remains in Software

The decoder (ConvTranspose2d + BatchNorm + skip connections) remains in software,
trained via backpropagation with Adam. It requires autograd and its weights are
updated every step — neither property is compatible with a fixed analog network.
The MS-SSIM+L1 loss computation stays in software — it is a multi-scale sequential
pipeline with no analog equivalent. The AdamState per-layer encoder weight updates,
Kolen-Pollack feedback head updates, and encoder forward pass (Conv2d + BN + ReLU)
all remain in software. These components together account for the remaining 23.7%
of step time and are not bottlenecks.

## (c) Interface Bandwidth Requirement

The accelerator receives per-layer error maps and returns feedback maps of the same
shape. For Layer 0 the transfer per call is:

    Input  (B, C, H, W) = (64, 64, 64, 64) × 4 bytes = 67,108,864 bytes
    Output (B, C, H, W) = (64, 64, 64, 64) × 4 bytes = 67,108,864 bytes
    Total per Layer 0 call = 134,217,728 bytes = 128 MB

All four layers per step:

    Layer 0: (64,  64, 64, 64) → 128.0 MB
    Layer 1: (64, 128, 32, 32) →  32.0 MB
    Layer 2: (64, 256, 16, 16) →  16.0 MB
    Layer 3: (64, 512,  8,  8) →  16.0 MB
    Total per step = 192.0 MB

At the realized 4.0× end-to-end speedup (1.18 steps/sec):

    Required BW = 192 MB × 1.18 = 226 MB/s

UCIe provides >100 GB/s — over 400× the required bandwidth. The interface is not
the bottleneck at any realistic operating point.

## (d) Bound Classification and Expected Change

On the current CPU the kernel is **compute-bound** (AI = 123.2 FLOP/byte vs CPU
ridge 13.5 FLOP/byte), achieving 651 GFLOP/s (47% of CPU peak). On the
memcapacitive network, the bound classification changes fundamentally: there is no
memory traffic for W_res or W_lat since they are physical device properties, and
the state s evolves in-place within the network without being written to or read
from any memory hierarchy. The bottleneck shifts to the **analog sampling rate** —
how fast the network state can be read out and the next input applied — which at
1 GHz gives 32,768 GFLOP/s equivalent throughput. This is a qualitatively
different bottleneck from any digital memory or compute bound: it is set by the
RC time constant of the memcapacitive devices and the ADC conversion rate, both
of which are physical properties of the fabrication process.
