# Milestone 2 — RC-DFA Memcapacitive Reservoir Accelerator

## ECE 510 Spring 2026

\---

## How to Reproduce M2 Simulations



Note: While the intent was to include XYCE and the analog crossbar simulation in M2, difficulties in getting the shim to operate and 

with the script still running 20hrs laters with no full output yet, it has been left for subsequent work.



### Simulator

**Icarus Verilog 12.0** (`iverilog` / `vvp`).

Install: `sudo apt-get install iverilog` (Ubuntu 24) or equivalent.

Verify: `iverilog -V` — should show version 12.0 or later.

No additional dependencies for the digital-only testbench runs below.
For Xyce co-simulation (full mixed-signal), see the co-simulation note at the end.

\---

### Compute Core Testbench

```bash
# From repository root
iverilog -g2012 \\
  -o project/m2/sim/tb\_compute\_core.vvp \\
  project/m2/tb/tb\_compute\_core.sv \\
  project/m2/rtl/compute\_core.sv

vvp project/m2/sim/tb\_compute\_core.vvp
```

Expected output (last line): `RESULT: PASS`

The testbench verifies 10 checks covering FSM sequencing, sample\_pulse count,
NODE\_IDX gather correctness, output range validity, done timing, and re-entrancy.

\---

### Interface Testbench

```bash
iverilog -g2012 \\
  -o project/m2/sim/tb\_interface.vvp \\
  project/m2/tb/tb\_interface.sv \\
  project/m2/rtl/interface.sv

vvp project/m2/sim/tb\_interface.vvp
```

Expected output (last line): `RESULT: PASS`

The testbench verifies 11 checks covering write transaction (4 AXI-S flits,
TLAST handshake, core\_start pulse, core\_e\_in assembly), read transaction
(m\_axis\_tvalid assertion, payload integrity, TLAST on final flit), post-TX
re-arming, and second-cycle re-entrancy.

\---

### Pre-processing (Reference Generation)

The Python script `scripts/gen\_reference.py` generates:

* `tb/tb\_reference\_values.svh` — FP32 reference hex include for testbenches
* `scripts/node\_idx\_check.txt` — human-readable NODE\_IDX table

Run before modifying testbenches or regenerating reference values:

```bash
cd project/m2/scripts
python3 gen\_reference.py   # requires numpy
```

Python dependencies: `numpy` (any recent version; tested with 1.26).

\---

### Waveform Generation

```bash
cd project/m2/scripts
python3 gen\_waveform.py   # requires matplotlib, numpy
```

Output: `project/m2/sim/waveform.png`

\---

## Deviations from M1 Plan

**No deviations.** The interface selection (UCIe), kernel scope
(`SpatialTiledReservoir.get\_feedback\_map`), and precision (FP32 at the digital
boundary) are unchanged from M1.

The RTL implements the digital compute core and UCIe AXI-S interface wrapper as
described in M1. The analog crossbar is present as `analog\_crossbar\_stub` — a
wire-only module with matching ports. In full co-simulation this stub is
overridden at runtime by the VPI shim (`vpi\_xyce\_shim.c`) that couples Icarus
to Xyce running the Verilog-A memcapacitive device model.

## Co-simulation Note (Xyce + Icarus)

Full mixed-signal co-simulation (Xyce + VPI shim) is the M3 deliverable.
For M2, the compute core testbench uses a behavioural analog model (deterministic
sinusoidal pattern) that drives `adc\_out` on each `sample\_pulse`. This validates
the digital FSM, NODE\_IDX gather, and output assembly independently of the analog
physics. The analog physics verification (Xyce transient, Verilog-A device model,
precision delta vs FP32) is documented in `precision.md` based on the Python
software model comparison.

