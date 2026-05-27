# Milestone 3 — RC-DFA Memcapacitive Reservoir Accelerator
## ECE 510 Spring 2026 — Mohammad Al-Mouhamed

---

## File Catalog

| Path | Description |
|---|---|
| `README.md` | This file. Catalogs all M3 files and reproduction instructions. |
| `rtl/top.sv` | Integrated top module instantiating interface_mod and compute_core with all inter-module signals wired. |
| `rtl/interface_mod.sv` | AXI-S UCIe interface module from M2 (unchanged). |
| `rtl/compute_core.sv` | RC reservoir compute core from M2 (unchanged). |
| `tb/tb_top.sv` | End-to-end co-simulation testbench. Drives DUT only through AXI-S interface ports. Compares output against FP32 Python reference. |
| `tb/tb_reference_m3.svh` | Auto-generated reference values (64 FP32 output channels) from `scripts/gen_reference_m3.py`. |
| `sim/cosim_run.log` | Co-simulation transcript. Final line: `RESULT: PASS`. |
| `sim/cosim_waveform.png` | End-to-end waveform annotated with write transaction, compute activity, and read transaction regions. |
| `analog/crossbar_rc.sp` | ngspice netlist for 128×128 memcapacitive crossbar full characterisation. |
| `analog/crossbar_rc.log` | ngspice transient output with measured t_10, t_90, t_99, tau, power. |
| `analog/analog_summary.txt` | Parsed analog metrics: timing, power, power density, routing. |
| `scripts/gen_reference_m3.py` | Generates `tb/tb_reference_m3.svh` — FP32 reservoir reference for testbench. |
| `scripts/extract_rc_timing.py` | Parses `analog/crossbar_rc.log` → writes `synth/analog_arrival.tcl` and `analog/analog_summary.txt`. |
| `synth/config.json` | OpenLane 2 configuration: sky130A, 10 ns clock, die 4500×4500 µm. |
| `synth/pnr_constraints.sdc` | SDC constraints used for PnR: clock definition, IO delays, analog arrival time. |
| `synth/analog_arrival.tcl` | Auto-generated input delay constraint from ngspice t_99 measurement. |
| `synth/openlane_run.log` | Full OpenLane 2 stdout/stderr from completed PnR run. |
| `synth/openlane_pnr.log` | Captured OpenLane step-level progress log. |
| `synth/timing_report.txt` | Post-route STA: WNS, TNS, setup/hold, critical path (nom_tt_025C_1v80). |
| `synth/area_report.txt` | Cell count by type, total area, flip-flop count (post-fill). |
| `synth/power_report.txt` | Post-route power: sequential, combinational, clock network, total (nom_tt_025C_1v80). |
| `synth/critical_path.md` | Critical path identification and analysis. |
| `synthesis_notes.md` | Narrative: what worked, what did not, scope status, M4 plan. |

---

## Co-Simulation Reproduction

**Simulator:** Icarus Verilog 12.0

**Install:**
```bash
sudo apt-get install iverilog
iverilog -V   # should show version 12.0
```

**Pre-processing (generate reference values):**
```bash
cd project/m3/scripts
python3 gen_reference_m3.py     # requires numpy
# outputs tb/tb_reference_m3.svh
```

**Run co-simulation:**
```bash
cd project/m3
mkdir -p sim
iverilog -g2012 -I tb/ \
  -o sim/tb_top.vvp \
  tb/tb_top.sv \
  rtl/top.sv \
  rtl/interface_mod.sv \
  rtl/compute_core.sv
vvp sim/tb_top.vvp | tee sim/cosim_run.log
```

**Expected output (last line):** `RESULT: PASS`

Python dependencies: `numpy` (tested 1.26). No other dependencies.

---

## OpenLane 2 Synthesis Reproduction

**OpenLane 2 version:** pip release `openlane==2.3.7`

**PDK:** sky130A via volare, hash `bdc9412b3e468c102d01b7cf6337be06ec6e9c9a`

**Install:**
```bash
python3 -m venv openlane_env
source openlane_env/bin/activate
pip install openlane==2.3.7
# PDK auto-installed on first run via volare
```

**Run (from project/m3/synth/):**
```bash
cd project/m3/synth
python3 -m openlane config.json
```

Or via Nix (reproducible environment):
```bash
nix --extra-experimental-features 'nix-command flakes' \
    run github:efabless/openlane2 -- config.json
```

**Environment variables required:**
```bash
export PDK_ROOT=~/.volare
export PDK=sky130A
```

**Configuration file:** `project/m3/synth/config.json`

**Completed run directory:** `synth/runs/RUN_2026-05-26_18-46-52/` (step 55, post-route STA complete)

**Tool versions used:**
- OpenLane 2.3.7 / OpenROAD (embedded)
- Yosys 0.33
- sky130A PDK (volare hash above)
- Python 3.12 on Ubuntu 24.04 (WSL2)
