# M3 Synthesis Plan
## ECE 510 Spring 2026 - Codefest 7 CLLM

Synthesis on compute_core.sv will be attempted by May 21, three days before the M3
deadline of May 24, leaving one iteration cycle to address any issues.

**Size:** compute_core.sv implements the FSM sequencing sample pulses over N_RES=128
reservoir nodes, a NODE_IDX gather over N_OUT=64 output channels, and 64 FP32
adc_out registers with no arithmetic datapath. I expect lower cell count and lower
area than the fallback result (1,422 cells, 15,187 um^2), with sequential elements
making up a higher fraction than the fallback's 13.45% since the 64-wide register
bank dominates rather than a combinational adder tree.

**Critical path location:** compute_core.sv performs no arithmetic in RTL, only
routing and index-gather. The critical path will be in the NODE_IDX gather MUX tree
selecting 64 values from 128 nodes, or in the FSM output fanout. Both are shallower
than the 20-stage carry-chain that produced an 8.337 ns arrival time and WNS of
-2.625 ns at the slow corner in the fallback.

**Precision:** compute_core.sv routes FP32 words (32-bit) from the ADC boundary with
no arithmetic, producing a cell mix dominated by mux and buffer cells rather than the
XOR/XNOR adder logic (xnor2_2: 153, xor2_2: 121) that dominated the fallback.

**Lessons learned from fallback:** the fanout-19 a31o at net _1216_ (0.507 ns delay)
caused 223 PnR-inserted timing repair buffers that still could not close the slow
corner (45 violations, TNS -62.1 ns). For compute_core.sv, OpenSTA fanout reports
will be checked before PnR and explicit RTL buffers inserted on any net above
fanout 10 before the design goes to the placer.
