# Precision and Data Format — RC-DFA Memcapacitive Reservoir Accelerator
## ECE 510 Spring 2026 — Project Milestone 2

---

## Numerical Format Choice

**Format: FP32 (IEEE 754 single-precision, 1 sign + 8 exponent + 23 mantissa bits)**

All data crossing the UCIe die-to-die interface is represented as FP32. This
applies to:

- Input error map words: `e_in[j]` for j = 0..N_IN−1 (64 words, Layer 0)
- Output feedback map words: `fb_out[j]` for j = 0..N_OUT−1 (64 words, Layer 0)
- ADC output registers in `compute_core.sv`: `adc_out[j]` for j = 0..N_RES−1

The analog network internals (node voltages `v_j(t)`, weight capacitances `C_ij`)
are continuous physical quantities and carry no fixed-point representation. They
are converted to FP32 at the ADC boundary (one ADC per reservoir node, 12-bit
physical ADC upsampled to 32-bit word via zero-extension and scaling in
`compute_core.sv`).

**Format is not FP16 or lower** because:

1. **Reservoir computing is inherently noise-tolerant** — the fixed random projection
   through W_res and W_lat provides built-in regularization, so precision errors in
   individual state values do not accumulate catastrophically across T=20 steps.
   Nevertheless, FP32 was retained at the interface because:
2. **The host software baseline is FP32** (PyTorch default). Reducing to FP16 at the
   interface would introduce a quantization step that is unnecessary given the
   available bandwidth: 226 MB/s at 4× speedup vs. UCIe >100 GB/s gives 400× headroom.
   There is no bandwidth motivation to reduce precision.
3. **The feedback map feeds a gradient computation** (`compute_weight_gradient` in
   `tiled14.py`). The outer-product dW = (fb_map ⊗ enc_output) / batch accumulates
   over batch_size=64 spatial locations per step. FP16 accumulation error could
   adversely affect encoder learning stability over many epochs; FP32 avoids this.

**If FP16 were chosen instead of FP32:** The error analysis below quantifies the
expected degradation. Mean absolute error of 1.24 × 10⁻⁴ per output channel is
within the noise tolerance of reservoir computing for the feedback map specifically,
but the accumulated effect on the gradient computation over 1000+ training steps
has not been characterized and poses a risk.

---

## Quantization Error Analysis

Analysis performed by `scripts/gen_reference.py` over 1000 randomly drawn spatial
locations (numpy.random.RandomState(9999), Layer 0 hyperparameters).

**Method:** FP32 reference (full-precision tanh reservoir, Python/numpy) vs. FP16
simulation (weights and activations cast to float16 at each step). The delta
represents the worst-case error introduced by an FP16 accelerator compared to the
current FP32 software baseline.

| Metric | FP16 vs FP32 |
|--------|-------------|
| Mean absolute error (per output channel) | 1.24 × 10⁻⁴ |
| Max absolute error (over all channels × all samples) | 9.78 × 10⁻⁴ |
| Std of absolute error | 1.11 × 10⁻⁴ |
| N samples | 1000 spatial locations × 64 channels = 64,000 comparisons |

**FP32 baseline absolute error (1 ULP):** 5.96 × 10⁻⁸ (machine epsilon for
single-precision mantissa).

The FP16 max error of 9.78 × 10⁻⁴ is approximately 16,400× the FP32 ULP. For the
reservoir feedback map specifically this is within the algorithm's noise tolerance
(reservoir computing was designed for noisy physical substrates), but it exceeds
the threshold we judge acceptable for the weight gradient accumulation path.

**The chosen FP32 format produces zero additional quantization error relative to
the software baseline** — the ADC→FP32 conversion is a lossless upcast from a
12-bit physical measurement, and the FP32 word format exactly matches the host
PyTorch tensor dtype.

---

## Acceptability Statement

**FP32 error is acceptable because:**

The digital boundary of the accelerator passes FP32 words that are bit-for-bit
identical to what the software baseline would pass (the ADC resolves to 12-bit
physical precision; the FP32 representation carries no additional error beyond
ADC quantization, which is 2⁻¹² ≈ 2.44 × 10⁻⁴ in the normalized [−1, +1]
output range). This is within the tolerance established by published reservoir
computing literature: Jaeger (2001) and Maass et al. (2002) both demonstrate
that echo-state networks retain useful computation under noise levels orders of
magnitude larger than FP32 rounding. The application-specific tolerance for
the feedback map is the convergence of the DFA encoder training loss, which has
been validated in the software baseline at this precision level (median epoch
85.12s, 3-epoch run, from `project_profile.txt`).

The FP32 threshold of ε ≤ 5.96 × 10⁻⁸ per word (1 ULP) is met by design: the
digital RTL performs no arithmetic on the feedback values — it only routes and
indexes them via the NODE_IDX gather — so no rounding occurs in `compute_core.sv`.

---

## Interface Word Width

Each FP32 word is 32 bits. The UCIe flit is 512 bits = 64 bytes = 16 FP32 words.
N_IN = N_OUT = 64 words → 4 flits per vector. This packs perfectly with zero
padding required, simplifying the TKEEP logic (always all-ones).

