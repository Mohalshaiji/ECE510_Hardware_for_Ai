# Project Scope Assessment
## ECE 510 Spring 2026 - Updated CF07

**Project:** RC-DFA Memcapacitive Reservoir Accelerator

## Original Scope

The project accelerates `SpatialTiledReservoir.get_feedback_map`, which accounts
for 76.3% of total training step time (6.091s out of 10.485s profiled across 3
batches). The kernel runs `s <- tanh(s @ W_res.T)` for T=20 recurrent steps over
B x H x W = 64 x 64 x 64 = 262,144 spatial locations per encoder layer. The
proposed hardware is a 128x128 memcapacitive crossbar network sampling at 1 GHz,
co-packaged with the host Intel Core Ultra 9 275HX via UCIe. W_res and W_lat are
encoded as device capacitances at fabrication. The interface transfers only the
input error map and output feedback map: 192 MB per training step across all four
encoder layers. Target kernel speedup is 50x, giving 4.0x end-to-end speedup by
Amdahl's law over the software baseline of 18.8 samples/sec (85.12s median epoch).

## M2 Status

compute_core.sv and interface.sv are implemented and verified in simulation via
Icarus Verilog. The compute core testbench passes all 10 checks covering FSM
sequencing, sample_pulse count, NODE_IDX gather correctness, output range, done
timing, and re-entrancy. The interface testbench passes all 11 checks covering
AXI-S write and read transactions, TLAST handshake, core_start pulse, and
re-arming. Precision is FP32 at the UCIe boundary, confirmed correct by the
quantization error analysis showing FP16 gradient accumulation error of 0.1% per
step vs 5e-5% under FP32. The analog crossbar is present as analog_crossbar_stub
pending co-simulation.

## Scope: Confirmed with one tool adjustment

The project scope is confirmed. Kernel target, hardware architecture, interface,
precision, and partition are all unchanged from M1 and M2.

The CF07 fallback synthesis on crossbar_mac closes timing at 100 MHz with +2.536 ns
setup slack and zero violations at the nominal corner (tt_025C_1v80), and +4.487 ns
at the fast corner (ff_n40C_1v95). This confirms the sky130 HD tool chain works and
that the 10 ns clock target is achievable for digital logic of this complexity.
LVS and DRC both pass clean. The slow corner (ss_100C_1v60) failed with WNS =
-2.625 ns and 45 violations, driven by a fanout-19 a31o net in the combinational
carry-chain accumulator. compute_core.sv has no multiply-accumulate datapath and no
carry-chain logic, so this class of failure is not expected to recur. The M3 plan
addresses this proactively by checking OpenSTA fanout reports before PnR and
inserting RTL buffers on any net above fanout 10.

compute_core.sv is expected to produce lower cell count and lower area than the
fallback (1,422 cells, 15,187 um^2 pre-PnR, 14,537.7 um^2 post-PnR standard cell
area, 81.9% utilization on a 150x150 um die), with a critical path in the NODE_IDX
gather MUX tree or FSM output fanout rather than a deep carry-chain. At 32-bit FP32
routing with no arithmetic, the cell mix will be mux and buffer dominated rather
than the XOR/XNOR adder logic (xnor2_2: 153, xor2_2: 121) that dominated the
fallback. Synthesis is targeted for May 21, three days before the M3 deadline.

## Scope Adjustment: Analog Co-simulation Tool

The Xyce transient run did not complete after 20 hours. For M3, the analog
co-simulation will move to ngspice using the same Verilog-A memcapacitive device
model. The RC settling time extracted from the ngspice transient will serve as the
system-level analog timing constraint fed into OpenSTA alongside the sky130 liberty
files, giving a complete system timing result: analog settling time plus digital
critical path from compute_core.sv synthesis. Analog power will be integrated from
the ngspice transient over one 1 GHz sampling cycle. Analog area cannot be produced
by any synthesis tool and will be estimated as 128 x 128 x unit cell area from the
Verilog-A device geometry parameters. If those parameters are insufficiently
specified for a physical area estimate, this will be documented explicitly as a
limitation with the derivation carried as far as the model allows. All other scope
items (kernel target, interface, precision, partition) are unchanged from M1 and M2.
