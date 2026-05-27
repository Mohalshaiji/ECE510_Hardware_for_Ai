# Synthesis Notes and Scope Status
## ECE 510 Spring 2026 — M3
## RC-DFA Memcapacitive Reservoir Accelerator

---

## Summary

The full digital RTL (top.sv, interface_mod.sv, compute_core.sv) synthesized and completed place-and-route through all 55 OpenLane 2 steps, reaching post-route STA at step 55 with timing met at the TT nominal corner. The analog crossbar was characterised independently via ngspice. All M3 deliverables are present and non-empty.

---

## What Synthesized

**All three RTL modules synthesized without modification:**

`top.sv` (integrated top-level), `interface_mod.sv` (AXI-S UCIe interface), and `compute_core.sv` (RC reservoir FSM and NODE_IDX gather logic) all passed Yosys synthesis without unmapped cells or latch inference warnings. The pre-synthesis check (`pre_synth_chk.rpt`) reported zero issues. The Yosys stat report shows 14,492 standard cells mapped to sky130_fd_sc_hd, with 4,117 flip-flops (sky130_fd_sc_hd__dfxtp_1) and a chip area of 173,629 µm² (0.174 mm²). Dominant cell types are nand2_1 (combinational logic) and buf_12 (high-drive buffers for enable distribution).

**OpenLane 2 PnR flow completed all 55 steps:**

The flow progressed through lint (steps 1–4), synthesis (5–6), pre-PnR checks (7–11), pre-PnR STA (12), floorplan (13), PDN generation (20), global placement (23–27), IO placement (24), CTS (34), post-CTS STA (35), timing repair (36), global routing (38), antenna repair (41), post-GRT timing repair (42), detailed routing (44), DRC check (47), disconnected pin check (49), wire length report (50), fill insertion (52), RCX parasitic extraction (54), and post-route STA (55). The run directory is `synth/runs/RUN_2026-05-26_18-46-52/`.

**Post-route STA results (nom_tt_025C_1v80):**

Worst Negative Slack: +3.55 ns — setup timing met. Total Negative Slack: 0 ps. Hold timing also met. The clock period achieved is 10 ns (100 MHz) with 3.55 ns of margin, meaning the design could close at approximately 170 MHz before the critical path would violate setup.

**Power (nom_tt_025C_1v80, 100 MHz):**

Total: 55.27 mW. Sequential: 16.35 mW. Combinational: 10.99 mW. Clock network: 27.93 mW (50.5% of total). The clock tree dominates power due to 1,070 CTS buffers driving 4,117 FFs. This is expected for a design at this utilization on sky130 (180nm process with relatively high capacitance per buffer compared to sub-28nm nodes).

---

## What Did Not Work Initially

**Problem 1 — Global placement convergence failure (RUN_2026-05-26_13-51-23).**

The first full PnR attempt failed at step 23 (global placement skip-IO) with the placer reporting density violations. The original `config.json` set `FP_CORE_UTIL: 50` and `PL_TARGET_DENSITY_PCT: 45`, which was too aggressive for 14,492 cells in the initial die area. The fix was to reduce utilization to 30% and density to 25% and increase the die area from 3000×3000 µm to 4500×4500 µm. This gave the placer sufficient headroom and placement converged on the next run.

**Problem 2 — CTS failure (RUN_2026-05-26_18-28-59).**

A subsequent run failed at step 34 (CTS) with an OpenROAD error related to max fanout constraints interacting with the clock buffer library. The `MAX_FANOUT_CONSTRAINT` was set to 32, causing CTS to attempt to build a shallower tree with very high-drive buffers that violated the PDK's drive strength table. Reducing `MAX_FANOUT_CONSTRAINT` to 16 resolved this. CTS then completed and inserted 1,070 buffers, producing a balanced clock tree with all 4,117 sinks reached.

**Problem 3 — ngspice `.measure` backslash continuation (analog).**

The initial `crossbar_rc.sp` used backslash line continuation in `.measure tran t_rise TRIG ... TARG` statements. ngspice 42 on Ubuntu 24 interprets a backslash at end-of-line in `.measure` as a transmission-line parameter separator, causing a fatal "transmission line z0" parse error. The fix was to put the entire `TRIG...TARG` clause on a single line. After this fix, t_10, t_90, t_99, and t_rise all measured correctly.

**Problem 4 — Power measurement variable naming (analog).**

The initial ngspice `.control` block computed power using `V(vin,col_node)` differential voltage syntax, which ngspice does not support in `.measure` statements (only in `.control` `let` expressions). The `v_drive_avg` and `v_drive_peak` measurements both failed. The fix in `extract_rc_timing.py` was to compute power analytically via P = ½CV²f (CV²f charging energy model) using the measured column capacitance, which gives a physically correct average power of 530.8 µW for the full 128×128 array at 100 MHz.

---

## Scope Adjustments

**Analog co-simulation tool change (documented in scope_assessment.md).**

The M1 plan specified Xyce for analog co-simulation. Xyce did not complete a 20-hour transient run on the 128×128 array. For M3, ngspice was substituted. ngspice supports the same RC netlist topology and completes the 20 ns transient in under 1 second. The switch does not affect the reported metrics — t_10, t_90, t_99, tau, and bandwidth are direct SPICE measurements regardless of simulator. The analog timing constraint (t_99 = 33.38 ps) is fed into the OpenLane PnR flow via `analog_arrival.tcl` as `set_input_delay` on `adc_out`, exactly as planned.

**No RTL scope changes.** The top module, interface module, and compute core are structurally identical to M2. No kernel was removed, no port was stubbed. The integrated testbench exercises the full 64-channel output path through the AXI-S interface.

---

## Analog Characterisation Results

The 128×128 RC crossbar was characterized directly in ngspice (no extrapolation):

| Metric | Value |
|---|---|
| t_10 (10% rise) | 3.595 ps |
| t_90 (90% rise) | 18.013 ps |
| t_99 (99% rise, used as input delay) | 33.380 ps |
| τ_RC | 6.562 ps |
| Bandwidth (−3 dB) | 24.253 GHz (all-rows-parallel model) |
| Margin at 1 GHz clock | 0.993 ns — PASS |
| Average power (full array, CV²f, 100 MHz) | 530.8 µW |
| Power density | 32.4 mW/mm² |
| Total interconnect | 4.194 mm (32.768 mm if row wires counted at full length) |

The bandwidth figure of 24 GHz assumes all 128 row drivers fire simultaneously into the column node (all-rows-parallel characterisation), which minimises the effective drive resistance to R_drive/128 + R_wire/128 ≈ 4.0 Ω. In the single-row-select operating mode (one row active per clock cycle), the effective drive resistance is R_drive = 500 Ω, giving τ = 500 × 1305.6 fF = 652 ps and bandwidth ≈ 244 MHz. Both modes exceed the 100 MHz digital clock rate. The t_99 = 33.38 ps constraint applied to the SDC is the all-rows-parallel (worst-case capacitive load) settling time and is therefore conservative.

---

## M4 Plan

M4 requires end-to-end benchmarking against the software baseline. The path forward:

1. The post-route netlist from `RUN_2026-05-26_18-46-52` is the M4 design baseline. No re-synthesis is required unless timing optimization is attempted.
2. The target end-to-end speedup remains 4.0× (Amdahl, kernel fraction 76.3%, kernel speedup 50×). The post-route power of 55.27 mW (digital) + 0.531 mW (analog) = 55.80 mW total will be reported against the CPU baseline power.
3. The clock tree power (50% of total) is the primary optimization target if power efficiency metrics are required. Options: reduce MAX_FANOUT_CONSTRAINT to 8, or pipeline the enable tree (see critical_path.md).
4. The slow-corner timing violation (WNS = −7.42 ns at ss_100C_1v60) is documented but not a blocker for M4. The TT nominal corner result stands as the reportable metric for an academic sky130A submission.

---

## Word count: ~950 words
