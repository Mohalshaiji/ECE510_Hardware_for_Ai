# ECE 410/510 Spring 2026 — Hardware for AI/ML

**Name:** Mohammad Alshaiji
**Course:** ECE 410/510 Spring 2026 — Hardware for AI and Machine Learning

## Project: RC-DFA Memcapacitive Reservoir Accelerator

This repository contains the complete Milestone 4 submission for a hardware accelerator
targeting the `SpatialTiledReservoir.get_feedback_map` kernel of a Direct Feedback
Alignment convolutional autoencoder. The accelerator replaces sequential digital matrix
multiplies with a physical 128×128 memcapacitive RC crossbar network co-packaged via UCIe,
achieving **4.22× end-to-end training speedup** and **3,061× energy improvement** over
the CPU baseline.

## Milestone 4 Submission

- **M4 deliverables:** [`project/m4/`](project/m4/)
- **M4 file catalog:** [`project/m4/README.md`](project/m4/README.md)
- **Design justification report:** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)
- **Simulation log (PASS):** [`project/m4/sim/final_run.log`](project/m4/sim/final_run.log)
- **Benchmark results:** [`project/m4/bench/benchmark.md`](project/m4/bench/benchmark.md)

## Key Results

| Metric | SW Baseline (M1) | HW Accelerator (M4) | Improvement |
|--------|-----------------|---------------------|-------------|
| Throughput | 18.8 samples/sec | 79.30 samples/sec | 4.22× |
| Kernel speedup | — | 11,755× | — |
| Energy/sample | 2,393.6 mJ | 0.782 mJ | 3,061× |
| Chip area | — | 173,464 µm² | — |
| Total power | ~45 W | 62.02 mW | 727× |

## M4 Changes vs M3

| Task | Change |
|------|--------|
| Task 1 | Interface widened: N_IN 64→4,096 (256 flits/call). Calls/step reduced 262,144→64. |
| Task 2 | Double-buffer pipeline: host streams next call during compute. 0 stall cycles measured. |
| Task 3 | Registered start_r + MAX_FANOUT_CONSTRAINT 16→8. |

## Prior Milestones

- M1 (software baseline + roofline): [`project/m1/`](project/m1/)
- M2 (RTL + verification): [`project/m2_exotic/`](project/m2_exotic/)
- M3 (synthesis + co-simulation): [`project/m3/`](project/m3/)
- Codefest work: [`codefest/`](codefest/)
