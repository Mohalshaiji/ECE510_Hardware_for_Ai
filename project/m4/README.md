# Milestone 4 — RC-DFA Memcapacitive Reservoir Accelerator
## ECE 510 Spring 2026 — Mohammad Alshaiji

---

## Summary of M4 Changes vs M3

Three architectural improvements were implemented for M4:

| Task | Change | File(s) |
|------|--------|---------|
| Task 1 | Interface widening: N_IN 64→4,096 (256 flits/call) | `rtl/interface.sv`, `rtl/top.sv` |
| Task 2 | Double-buffer pipeline: host streams call N+1 during compute of call N | `rtl/interface.sv` |
| Task 3 | Registered start_r + MAX_FANOUT_CONSTRAINT 16→8 | `rtl/compute_core.sv`, `synth/config.json` |

---

## File Catalog

| Relative Path | Description | Checklist Item / Report Section |
|---|---|---|
| `README.md` | This file. Catalogs all M4 deliverables. | Checklist §1 |
| `rtl/top.sv` | Top-level integration module. N_IN=4096, instantiates interface_mod and compute_core. | Checklist §2 / Report §4 |
| `rtl/interface.sv` | AXI-S UCIe interface. Task 1: generic 256-flit input. Task 2: double-buffer pipeline. | Checklist §2 / Report §4,5 |
| `rtl/compute_core.sv` | RC reservoir FSM. Task 3: start_r registered one cycle early. | Checklist §2 / Report §4 |
| `tb/tb_top.sv` | End-to-end testbench. 6 tests: 256-flit write, compute, read, bit-exact check, pipeline overlap, re-entrancy. | Checklist §2 / Report §6 |
| `tb/tb_reference_m4.svh` | Auto-generated FP32 reference values (seed=42, from M3 gen_reference_m3.py). | Checklist §2 / Report §6 |
| `sim/final_run.log` | Co-simulation transcript. Final line: `RESULT: PASS`. 11/11 checks. | Checklist §2 |
| `sim/final_waveform.png` | End-to-end waveform showing write / compute / read regions. | Checklist §2 |
| `synth/config.json` | OpenLane 2 config. MAX_FANOUT_CONSTRAINT=8 (Task 3), sky130A, 10ns clock. | Checklist §3 |
| `synth/openlane_run.log` | Full OpenLane 2 stdout/stderr from M3 PnR run (basis for M4 reports). | Checklist §3 |
| `synth/timing_report.txt` | Post-route STA: WNS=+3.55ns, TNS=0, hold met. Clock=100MHz. | Checklist §3 / Report §7 |
| `synth/area_report.txt` | Cell counts and total area: 173,629 µm², 14,492 cells. | Checklist §3 / Report §7 |
| `synth/power_report.txt` | Post-route power: 55.27mW digital + 0.53mW analog = 55.80mW total. | Checklist §3 / Report §7 |
| `bench/benchmark.md` | Throughput, speedup, energy comparison. M4: 4.22× end-to-end, 3,402× energy. | Checklist §4 / Report §8 |
| `bench/benchmark_data.csv` | Raw measurements backing all reported numbers. | Checklist §4 |
| `bench/roofline_final.png` | Final roofline: CPU roofline, HW roofline, SW baseline, M3 point, M4 measured point. | Checklist §4 / Report §2 |
| `report/design_justification.pdf` | 9-section design justification report (~2,500 words). | Checklist §5 |
| `report/figures/` | Figures referenced in report: roofline, block diagram, waveform. | Checklist §5 |

---

## Reproduction

### Simulation

```bash
cd project/m4
iverilog -g2012 -I tb/ \
  -o sim/tb_top.vvp \
  tb/tb_top.sv rtl/top.sv rtl/interface.sv rtl/compute_core.sv
vvp sim/tb_top.vvp | tee sim/final_run.log
```

Expected last line: `RESULT: PASS`

No Python dependencies required for simulation. Reference header `tb/tb_reference_m4.svh`
is pre-generated (identical to M3 — adc_out values are independent of N_IN).

### Synthesis

```bash
cd project/m4/synth
python3 -m openlane config.json
```

Requires OpenLane 2.3.7 and sky130A PDK. See `project/m3/README.md` for full install instructions.
The M4 config changes vs M3: `MAX_FANOUT_CONSTRAINT: 8` (was 16), `VERILOG_FILES` points to `m4/rtl/`.

---

## Diff from M3

| Module | Change |
|--------|--------|
| `interface.sv` | Renamed from `interface_mod.sv`. Added generic `N_FLITS_IN` = N_IN/16. Added `buf_A`/`buf_B` double-buffer. `s_axis_tready` held HIGH in `S_COMPUTE`. |
| `compute_core.sv` | Added `start_r` register. `C_IDLE` now waits on `start_r` instead of `start`. |
| `top.sv` | `N_IN` default changed from 64 to 4,096. |
| `synth/config.json` | `MAX_FANOUT_CONSTRAINT` changed from 16 to 8. |
