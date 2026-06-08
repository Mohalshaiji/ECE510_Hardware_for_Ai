# Benchmark Comparison — M4
## RC-DFA Memcapacitive Reservoir Accelerator
## ECE 510 Spring 2026 — Mohammad Alshaiji

---

## Platform Descriptions

**Software baseline:** Intel Core Ultra 9 275HX, 24 threads, Windows 10 (26200),
PyTorch 2.12.0.dev20260407+cu128 forced to CPU. Source: `project/m1/sw_baseline.md`,
reproduced by running `tiled_profile.py` with `N_EPOCHS=10`.

**Hardware accelerator (M4):** 128×128 memcapacitive RC crossbar (1 GHz analog
sampling) co-simulated with sky130A digital interface FSM (100 MHz). Three
architectural improvements relative to M3:

- **Task 1 — Interface widening:** N_IN expanded from 64 to 4,096 channels.
  Input payload per call is now 4,096 × 32b = 131,072 b = **256 × 512-bit flits**
  (was 4 flits). Calls per training step reduced from 262,144 to 64 for Layer 0.
- **Task 2 — Double-buffer pipeline:** `s_axis_tready` held HIGH during `S_COMPUTE`
  via a second e_in register bank. The host streams call N+1 while call N computes.
  Measured stall cycles during overlap: **0** (verified by `tb_top.sv` TEST 5).
- **Task 3 — Registered start_r:** `start` signal registered one cycle before
  consumption. `config.json` sets `MAX_FANOUT_CONSTRAINT: 8` (was 16).

Simulation via Icarus Verilog 12.0. All throughput figures derived from
cycle-accurate simulation. Power from OpenLane 2.3.7 post-route analysis,
`RUN_2026-06-07_16-54-00`, sky130A nom_tt_025C_1v80.

---

## Cycle-Count Results (M4)

Cycle formula (M4): `cycles/call = N_FLITS_IN + 1 + 2T + 1 + N_FLITS_OUT`
= 256 (write) + 1 (start_r) + 2T (compute) + 1 (C_DONE) + 4 (send).

With pipeline overlap (Task 2), effective cycles/call =
`max(N_FLITS_IN, 1 + 2T + 1) + N_FLITS_OUT` = max(256, 42) + 4 = **260 cycles**.

| Layer | out_ch | T  | Calls/step | Cycles/call | ns/call   | HW time/step |
|-------|--------|----|-----------|-------------|-----------|--------------|
| 0     | 64     | 20 | 64        | 260 (pipe)  | 2,600 ns  | 0.1664 ms    |
| 1     | 128    | 20 | 16        | 260 (pipe)  | 2,600 ns  | 0.0416 ms    |
| 2     | 256    | 15 | 4         | 260 (pipe)  | 2,600 ns  | 0.0104 ms    |
| 3     | 512    | 10 | 1         | 260 (pipe)  | 2,600 ns  | 0.0026 ms    |

**Total HW kernel time/step: 0.2210 ms** (sum of all layers, pipelined)

---

## Benchmark Table

| Metric                         | SW Baseline (M1)    | HW Accelerator (M4)    | Speedup     |
|-------------------------------|---------------------|------------------------|-------------|
| Epoch time                    | 85.12 s             | 20.19 s                | **4.22×**   |
| Step time                     | 3,405 ms            | 807.2 ms               | **4.22×**   |
| Training throughput           | 18.8 samples/sec    | 79.30 samples/sec      | **4.22×**   |
| Kernel time/step (all layers) | 2,598 ms            | 0.221 ms               | **11,755×** |
| Calls/step (Layer 0)          | 1 (batched SW call) | 64                     | —           |
| Cycles/call (Layer 0, pipe)   | N/A                 | 260                    | —           |
| Active compute power          | ~45 W               | 62.02 mW               | —           |
| Energy per sample             | 2,393.6 mJ          | 0.782 mJ               | **3,061×**  |

Power breakdown (nom_tt_025C_1v80, post-route):
Sequential 18.52 mW (30.1%) + Combinational 11.89 mW (19.3%) +
Clock 31.08 mW (50.5%) + Analog crossbar 0.53 mW = **62.02 mW total**.

Note on Task 3: MAX_FANOUT_CONSTRAINT: 8 did not reduce clock tree power as
expected. Clock power remained at 50.5% because Task 2's double-buffer added
~4,096 new flip-flops (buf_A and buf_B, 131,072 bits each), increasing total
FF count from ~4,117 (M3) to ~8,192+ (M4), which dominates CTS regardless
of fanout constraint. See Section 9 of the design justification report.

---

## Speedup Derivation

**SW non-kernel time/step:** 3,405 × (1 − 0.763) = 807.0 ms (unchanged)

**HW kernel time/step:** 0.221 ms (pipelined, all 4 layers)

**HW total step time:** 0.221 + 807.0 = **807.2 ms**

**End-to-end speedup:** 3,405 / 807.2 = **4.22×**

**Amdahl check:** 1 / (0.237 + 0.763 / 11755) ≈ **4.22×** ✓

**Comparison to M3:** M3 achieved 3.42× (kernel speedup 13.7×). M4 improves to
4.22× (kernel speedup 11,755×) by reducing calls/step from 262,144 to 64 (Task 1).
The end-to-end gain is Amdahl-limited by the non-kernel portion (23.7%).

---

## Energy Efficiency

**SW:** 45 W / 18.8 samples/sec = 2,393.6 mJ/sample

**HW:** 62.02 mW / 79.30 samples/sec = **0.782 mJ/sample**

**Energy improvement: 3,061×**

---

## Measurement Method

All cycle counts from Icarus Verilog 12.0 simulation at 100 MHz with full
AXI-S protocol exercised end-to-end (`sim/final_run.log`). Pipeline overlap
TEST 5 confirmed 0 stall cycles. Power from OpenLane 2.3.7 post-route
`report_power` at nom_tt_025C_1v80. Raw data: `bench/benchmark_data.csv`.
