# Heilmeier Questions — RC-DFA VAE Project
## ECE 510 Spring 2026 — Updated with Profiling Data (M1)

---

## Q1: What are you trying to do?

We are building a hardware-accelerated convolutional autoencoder that trains using
**Direct Feedback Alignment with a Spatial Tiled Reservoir (RC-DFA)** instead of
backpropagation. The encoder learns via fixed random reservoir echo-state networks
that project a global reconstruction error signal back to local weight gradients at
each layer — no weight transport, no backward pass through the network. The target
application is satellite image compression (4-channel RGBD patches, 128×128,
achieving 64:1 compression to a 16×8×8 latent space). The kernel to be accelerated
is `SpatialTiledReservoir.get_feedback_map`: the recurrent echo-state dynamics
`s ← tanh(s @ W_res.T)` applied T=20 times over all spatial locations of each
encoder layer's error map, which accounts for 76.3% of training step time.

---

## Q2: What is done today and what are the limits?

Current software training on an Intel Core Ultra 9 275HX CPU runs at **18.8
samples/sec** (median epoch time 85.12s over 3 epochs, 25 steps/epoch, batch=64).
Profiling with cProfile over the first 3 batches identifies
`SpatialTiledReservoir.get_feedback_map` as the dominant bottleneck:

- **76.3% of total step time** (6.091s tottime / 11.095s total, from
  `codefest/cf02/profiling/project_profile.txt`)
- Observed throughput: **651 GFLOP/s** per reservoir call
- CPU theoretical peak FP32 (AVX2): 1,382 GFLOP/s — utilisation 47%

The kernel's arithmetic intensity is **123.2 FLOP/byte**, placing it compute-bound
above the CPU ridge point of 13.5 FLOP/byte. The fundamental limit is that the
software simulation discretises what is inherently a continuous recurrent dynamical
system into T=20 sequential digital matrix multiplies. Each step depends on the
previous, so no step-level parallelism is possible in software. The reservoir
dynamics — fading memory, nonlinear state evolution, spatial mixing — are being
approximated digitally at significant cost when a physical analog system could
exhibit those same dynamics natively.

---

## Q3: What is your approach and why is it better?

We propose replacing the software reservoir simulation with a **physical analog
reservoir implemented as a memcapacitive crossbar network**, co-packaged with the
host CPU via UCIe. Rather than computing `s ← tanh(s @ W_res.T)` as a matrix
multiply, the crossbar network evolves as a physical dynamical system — voltage
inputs drive charge redistribution through the capacitive network, and the network
state s evolves continuously according to its own dynamics. The T=20 timesteps
become T samples of a physical process rather than T sequential digital operations.

This is the natural hardware substrate for reservoir computing specifically because:
(1) reservoir computing was originally conceived for physical dynamical systems,
not digital simulation; (2) the fixed random weights W_res and W_lat are set once
at fabrication as device capacitances — no loading, no memory traffic, no precision
requirement since reservoirs are noise-tolerant by design; (3) the nonlinearity
tanh is implemented by the device physics of the memcapacitive elements rather than
as an explicit digital operation.

From the profiling data, the software reservoir consumes 76.3% of step time at
651 GFLOP/s equivalent. A 128×128 memcapacitive network sampling at 1 GHz achieves
32,768 GFLOP/s equivalent throughput, giving a **50× kernel speedup** and **4.0×
end-to-end training speedup** by Amdahl's law. More importantly, the physical
implementation eliminates the conceptual mismatch between the algorithm (a
continuous dynamical system) and its substrate (sequential digital arithmetic).
