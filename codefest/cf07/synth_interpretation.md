# Synthesis Interpretation - crossbar_mac (CF06 Fallback)
## ECE 510 Spring 2026 - Codefest 7 CLLM

**Target:** sky130 HD, OpenLane 2 full PnR. Clock 10.0 ns, die 150x150 um absolute,
PL_TARGET_DENSITY_PCT = 90. Run: RUN_2026-05-18_03-59-58.

## Clock and Slack

Closes at 100 MHz at nominal (tt_025C_1v80, setup WS +2.536 ns) and fast
(ff_n40C_1v95, setup WS +4.487 ns) with zero violations. Fails the slow corner
(ss_100C_1v60, setup WS -2.625 ns, TNS -62.1 ns, 45 violations). The design would
not be manufacturable across process variation at this clock without fixing the slow
corner. Hold clean at all corners.

## Critical Path

Worst path: `data_in[9]` to `_2702_` (accum_out[40]), arrival 8.337 ns, required
9.664 ns. The MAC array is fully unrolled with no pipeline registers, forcing the
entire multiply-accumulate through one clock cycle. Bottleneck is a31o at net
`_1216_`, fanout 19, delay 0.507 ns. The PnR tool inserted 223 timing repair
buffers to address this and still could not close the slow corner. A pipeline
register between partial sum and accumulator would cut the critical path in half.

## Area

Pre-PnR: 1,422 cells, 15,187 um^2. Post-PnR: 1,906 instances, cell area 14,537.7
um^2, core 17,759.5 um^2, die 22,500 um^2, utilization 81.9% vs 90% target. Top
three by count: xnor2_2 (153), nand2_2 (130), xor2_2 (121).

## Warnings and Flags

ss_100C_1v60 fails with WNS = -2.625 ns, TNS = -62.1 ns, 45 setup violations.
The max RC variant of the same corner produces identical numbers; the min RC variant
has 44 violations, WNS = -2.336 ns, TNS = -55.1 ns. No hold violations at any
corner. 5 fanout violations at all corners, all on net `_1216_` (fanout 19, a31o).
Yosys warning at log line 23: `\partial` inferred as registers rather than memory,
producing 96 dfxtp_2 flip-flops. 7 unannotated nets post-route, 0 affect timing.
DRC: 0 errors (Magic and KLayout). LVS: 0 mismatches. IR drop: 2.08 mV VPWR.
