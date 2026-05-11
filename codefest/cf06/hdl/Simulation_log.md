VCD info: dumpfile crossbar_tb.vcd opened for output.
=== CF-06  Binary-Weight Crossbar MAC Testbench ===
-- TC_CF06: CF-06 specified weight matrix ----------
Weights: row0=[+1,-1,+1,-1]  row1=[+1,+1,-1,-1]
row2=[-1,+1,+1,-1]  row3=[-1,-1,-1,+1]
Input  : [10, 20, 30, 40]
Expected: out[0]=-40  out[1]=0  out[2]=-20  out[3]=-20
[PASS]          TC_CF06  out[0] = -40
[PASS]          TC_CF06  out[1] = 0
[PASS]          TC_CF06  out[2] = -20
[PASS]          TC_CF06  out[3] = -20
-> all outputs match
-- TC_ALL_P: All +1 weights ------------------------
[PASS]         TC_ALL_P  out[0] = 100
[PASS]         TC_ALL_P  out[1] = 100
[PASS]         TC_ALL_P  out[2] = 100
[PASS]         TC_ALL_P  out[3] = 100
-> all outputs match
-- TC_ALL_N: All -1 weights ------------------------
[PASS]         TC_ALL_N  out[0] = -100
[PASS]         TC_ALL_N  out[1] = -100
[PASS]         TC_ALL_N  out[2] = -100
[PASS]         TC_ALL_N  out[3] = -100
-> all outputs match
-- TC_CHESS: Checkerboard weights ------------------
[PASS]         TC_CHESS  out[0] = -2
[PASS]         TC_CHESS  out[1] = 2
[PASS]         TC_CHESS  out[2] = -2
[PASS]         TC_CHESS  out[3] = 2
-> all outputs match
-- TC_MULTI: 3-cycle accumulation ------------------
[PASS]         TC_MULTI  out[0] = -120
[PASS]         TC_MULTI  out[1] = 0
[PASS]         TC_MULTI  out[2] = -60
[PASS]         TC_MULTI  out[3] = -60
-> all outputs match
-- TC_CLR: accum_clr mid-stream --------------------
[PASS]           TC_CLR  out[0] = -40
[PASS]           TC_CLR  out[1] = 0
[PASS]           TC_CLR  out[2] = -20
[PASS]           TC_CLR  out[3] = -20
-> all outputs match
-- TC_BOUND: Boundary inputs (+127,+127,-128,-128) -
[PASS]         TC_BOUND  out[0] = -2
[PASS]         TC_BOUND  out[1] = -2
[PASS]         TC_BOUND  out[2] = -2
[PASS]         TC_BOUND  out[3] = -2
-> all outputs match
=== RESULTS: 28 passed   0 failed   28 total ===
=== STATUS : ALL TESTS PASSED ===
crossbar_tb.sv:316: $finish called at 1426000 (1ps)
