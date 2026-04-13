# Interface Selection — RC-DFA Memcapacitive Accelerator
## ECE 510 Spring 2026 — Project Milestone 1

## Host Platform

The assumed host platform is a **laptop CPU SoC** (Intel Core Ultra 9 275HX),
communicating with a co-packaged memcapacitive chiplet via UCIe. The host handles
the encoder forward pass, decoder BP, loss, and optimizer steps in software. The
chiplet implements `SpatialTiledReservoir.get_feedback_map` for all four encoder
layers as a physical analog reservoir — a 128×128 memcapacitive crossbar network
sampling at 1 GHz. W_res and W_lat are encoded as device capacitances at
fabrication; only the input error map and output feedback map cross the interface.

## Interface Selected: UCIe (Universal Chiplet Interconnect Express)

UCIe is selected from the project interface table.

## Bandwidth Requirement Calculation

The only data crossing the interface per call is the input error map and the output
feedback map. W_res, W_lat, and the reservoir state s are all internal to the
physical network and never transferred.

    Layer 0 per call: (B=64, C=64, H=64, W=64) × 4 bytes × 2 (in+out)
                    = 134,217,728 bytes = 128 MB

All four layers per step:

    Layer 0: (64,  64, 64, 64) → 128.0 MB
    Layer 1: (64, 128, 32, 32) →  32.0 MB
    Layer 2: (64, 256, 16, 16) →  16.0 MB
    Layer 3: (64, 512,  8,  8) →  16.0 MB
    Total per step = 192.0 MB

At current SW throughput (0.294 steps/sec):

    Required BW = 192 MB × 0.294 = 56.4 MB/s

At realized 4.0× end-to-end speedup (1.18 steps/sec):

    Required BW = 192 MB × 1.18 = 226 MB/s

At theoretical maximum if full step accelerated (294 steps/sec):

    Required BW = 192 MB × 294 = 56.4 GB/s

## Interface Rated Bandwidth vs Required

| Interface | Rated BW | Required (4× end-to-end) | Bottleneck? |
|-----------|----------|--------------------------|-------------|
| SPI | 0.008 GB/s | 0.226 GB/s | Yes — 28× too slow |
| I²C | 0.0004 GB/s | 0.226 GB/s | Yes — completely inadequate |
| AXI4-Lite | ~0.5 GB/s | 0.226 GB/s | No — marginal, no headroom |
| AXI4 Stream | ~12.8 GB/s | 0.226 GB/s | No — 57× headroom |
| PCIe 5.0 ×16 | ~64 GB/s | 0.226 GB/s | No — ample headroom |
| **UCIe** | **>100 GB/s** | **0.226 GB/s** | **No — >400× headroom** |

UCIe is selected because **latency, not bandwidth, is the binding constraint**.
Each training step issues four sequential reservoir calls (one per encoder layer),
each requiring a round-trip transfer. PCIe controller overhead (~1 µs per
transaction) would add ~4 µs per step — negligible at current throughput but a
hard floor that limits future scaling. UCIe latency is sub-nanosecond, matching
the 1 GHz sampling rate of the analog network. UCIe is also the natural physical
interface for a 2.5D co-packaged chiplet die alongside the host CPU.

## Bottleneck Status on Roofline

The design is not interface-bound. With W_res, W_lat, and state s all internal
to the physical network, the effective AI rises to 2,464 FLOP/byte — the kernel
is compute-bound on the chiplet and hits the 32,768 GFLOP/s ceiling. The binding
bottleneck is the **analog sampling rate**: the RC time constant of the
memcapacitive devices and the ADC conversion rate at readout, both set by
fabrication process, not by the die-to-die interconnect.
