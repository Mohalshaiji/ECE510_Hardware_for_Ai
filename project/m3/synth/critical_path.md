# Critical Path Analysis — RC-DFA Memcapacitive Reservoir Accelerator
## ECE 510 Spring 2026 — M3 Post-Route STA (nom_tt_025C_1v80)

---

## Critical Path Summary

**Corner:** nom_tt_025C_1v80 (nominal process, 25°C, 1.8V)
**Clock period:** 10.000 ns (100 MHz)
**Post-route WNS (setup):** +3.55 ns — **TIMING MET**
**Critical path data arrival:** 6.45 ns
**Clock arrival at endpoint:** ~10.0 ns
**Setup slack:** +3.55 ns

---

## Path Identification

**Start point:** `u_iface/state_reg[3]/CLK` (flip-flop clock pin, interface FSM state register)

**End point:** `u_core/fb_r_reg[2][63]/D` (flip-flop data input, feedback register bit 63 of channel 2)

**Logic stages on critical path (post-CTS, post-route):**

1. **FSM state register output** — `u_iface/state_reg[3]/Q` launches at clock edge. The state register encodes the AXI-S interface FSM (IDLE → WRITE → COMPUTE → READ states).

2. **Enable decode logic** — A fanout-16 enable tree (`sky130_fd_sc_hd__buf_12` chain) decodes the FSM state to generate `core_start` and channel select signals. This is the dominant stage: 8 levels of buffering added by CTS to drive 4,117 flip-flops from a single enable source.

3. **NODE_IDX gather mux** — A 128:1 multiplexer tree in `compute_core` selects the appropriate reservoir node output based on `node_idx`. This is a log2(128) = 7 level mux tree synthesized as `sky130_fd_sc_hd__mux2_1` chains.

4. **Output register setup** — `u_core/fb_r_reg[2][63]/D` is the data input to the feedback output register. Setup time is ~70 ps at TT corner.

**Total combinational delay:** ~6.45 ns (3.55 ns slack against 10 ns clock)

---

## Why This Is the Critical Path

The path is dominated by two factors. First, the enable distribution tree: a single `core_start` signal must reach all 4,117 flip-flops within one clock cycle. CTS inserted 1,070 buffers to balance this tree, but the deepest branch still presents 8 buffer stages of delay (~2.5 ns cumulative at TT). Second, the 128:1 NODE_IDX mux tree adds ~3.0 ns of combinational delay from the mux select inputs to the selected reservoir node output.

The path does not go through any arithmetic — the compute core performs no multiplication or addition on the critical path. The bottleneck is purely structural: a high-fanout control signal feeding a wide mux.

---

## What Would Shorten It

**Option 1 — Register the enable tree mid-point.** Adding a pipeline register between the FSM state decode and the NODE_IDX mux would split the 6.45 ns path into two ~3.2 ns stages, easily meeting timing at 200 MHz. This would add one cycle of latency to the compute phase but would not affect throughput since the compute phase runs for many cycles.

**Option 2 — Reduce fanout at synthesis.** The current `MAX_FANOUT_CONSTRAINT: 16` in `config.json` was set conservatively. Reducing to 8 would cause Yosys to insert more buffer levels during synthesis, giving the CTS tool a flatter starting tree. This may reduce the CTS buffer count and clock power (currently 50% of total at 27.93 mW).

**Option 3 — Widen the NODE_IDX mux.** The current 128-node mux is purely combinational. Encoding NODE_IDX as a one-hot select and using a wider mux primitive would reduce the tree depth from 7 levels to 2, saving ~1.5 ns on that stage alone.

For M4, Option 1 (pipeline register) is the lowest-risk change and would allow a clock target of 200 MHz without redesigning the datapath.

---

## Slow Corner Note

At the `nom_ss_100C_1v60` (slow corner, 100°C, 1.6V), WNS = −7.42 ns. This corner is not the signoff corner for an educational sky130A design — the nominal TT corner is standard for academic ECE course submissions — but the magnitude indicates the enable tree and mux path are sensitive to process variation. A production-quality design would require multi-corner closure; for M4 the TT corner result stands as the reportable metric.
