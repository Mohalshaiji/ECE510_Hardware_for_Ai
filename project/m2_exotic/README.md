# Milestone 2 — RC-DFA Memcapacitive Reservoir Accelerator

## ECE 510 Spring 2026

---

## How to Reproduce M2 Simulations

Note: While the intent was to include XYCE and the analog crossbar simulation in M2, difficulties in getting the shim to operate and
with the script still running 20hrs laters with no full output yet, it has been left for subsequent work.

### Simulator

**Icarus Verilog 12.0** (`iverilog` / `vvp`).

Install: `sudo apt-get install iverilog` (Ubuntu 24) or equivalent.

Verify: `iverilog -V` — should show version 12.0 or later.

No additional dependencies for the digital-only testbench runs below.
For Xyce co-simulation (full mixed-signal), see the co-simulation note at the end.

---

### Pre-processing (Reference Generation)

Run this first, before compiling or running the testbenches. The compute core
testbench includes `tb_reference_values.svh`, which this script generates.

```bash
cd project/m2/scripts
python3 gen_reference.py        # requires numpy
cp tb_reference_values.svh ../tb/
```

Python dependencies: `numpy` (any recent version; tested with 1.26).

Outputs:
- `tb/tb_reference_values.svh` — FP32 reference hex include for testbenches
- `node_idx_check.txt` — human-readable NODE_IDX table (written to `scripts/`)

---

### Compute Core Testbench

All commands run from `project/m2/`.

```bash
mkdir -p sim
iverilog -g2012 -I tb/ \
  -o sim/tb_compute_core.vvp \
  tb/tb_compute_core.sv \
  rtl/compute_core.sv

vvp sim/tb_compute_core.vvp
```

Expected output (last line): `RESULT: PASS`

The testbench verifies 10 checks covering FSM sequencing, sample_pulse count,
NODE_IDX gather correctness, output range validity, done timing, and re-entrancy.

---

### Interface Testbench

All commands run from `project/m2/`.

```bash
mkdir -p sim
iverilog -g2012 \
  -o sim/tb_interface.vvp \
  tb/tb_interface.sv \
  rtl/interface.sv

vvp sim/tb_interface.vvp
```

Expected output (last line): `RESULT: PASS`

The testbench verifies 11 checks covering write transaction (4 AXI-S flits,
TLAST handshake, core_start pulse, core_e_in assembly), read transaction
(m_axis_tvalid assertion, payload integrity, TLAST on final flit), post-TX
re-arming, and second-cycle re-entrancy.

---

### Waveform

`sim/waveform.png` is committed directly to the repository. It was generated
from the compute core VCD output and requires no additional steps to reproduce.

---

## Deviations from M1 Plan

**No deviations.** The interface selection (UCIe), kernel scope
(`SpatialTiledReservoir.get_feedback_map`), and precision (FP32 at the digital
boundary) are unchanged from M1.

The RTL implements the digital compute core and UCIe AXI-S interface wrapper as
described in M1. The analog crossbar is present as `analog_crossbar_stub` — a
wire-only module with matching ports. In full co-simulation this stub is
overridden at runtime by the VPI shim (`vpi_xyce_shim.c`) that couples Icarus
to Xyce running the Verilog-A memcapacitive device model.

## Co-simulation Note (Xyce + Icarus)

Full mixed-signal co-simulation (Xyce + VPI shim) is the M3 deliverable.
For M2, the compute core testbench uses a behavioural analog model (deterministic
sinusoidal pattern) that drives `adc_out` on each `sample_pulse`. This validates
the digital FSM, NODE_IDX gather, and output assembly independently of the analog
physics. The analog physics verification (Xyce transient, Verilog-A device model,
precision delta vs FP32) is documented in `precision.md` based on the Python
software model comparison.
