# Benchmark Results
## Codefest 9 | ECE 410/510 | Spring 2026
## RC-DFA Memcapacitive Reservoir Accelerator

## Platform Descriptions

Software baseline: Intel Core Ultra 9 275HX, 24 threads, Windows 10 (26200), PyTorch 2.12.0.dev20260407+cu128 forced to CPU. Source: project/m1/sw_baseline.md, reproduced by running tiled_profile.py with N_EPOCHS=10.

Hardware accelerator: 128x128 memcapacitive RC crossbar (1 GHz analog sampling) co-simulated with sky130A digital interface FSM (100 MHz). Simulation via Icarus Verilog 12.0. The throughput testbench tb_throughput.sv (codefest/cf09/tb/tb_throughput.sv) runs 100 sequential reservoir calls and counts clock cycles. All HW figures in this table are measured from that simulation.

## Cycle-Accurate Simulation Results

The DUT processes one spatial location (64 FP32 channels) per call. The training kernel requires one call per spatial location in the batch-spatially-unrolled error map. Layer 0 has SP = 64 x 64 x 64 = 262,144 spatial locations per step; layers 1, 2, 3 have SP = 65,536, 16,384, and 4,096 respectively.

Simulated cycle counts (100 sequential calls each, consistent across all calls):

| Layer | out_ch | H  | W  | T  | SP      | cycles/call | ns/call | calls/step | HW time/step |
|-------|--------|----|----|----|---------|-------------|---------|------------|--------------|
| 0     | 64     | 64 | 64 | 20 | 262,144 | 55          | 550 ns  | 262,144    | 144.2 ms     |
| 1     | 128    | 32 | 32 | 20 | 65,536  | 55          | 550 ns  | 65,536     | 36.0 ms      |
| 2     | 256    | 16 | 16 | 15 | 16,384  | 45          | 450 ns  | 16,384     | 7.4 ms       |
| 3     | 512    | 8  | 8  | 10 | 4,096   | 35          | 350 ns  | 4,096      | 1.4 ms       |

Cycle count formula (confirmed by simulation): cycles = 2T + 15. At T=20: 55, T=15: 45, T=10: 35.

Breakdown for T=20: 4 cycles write (flit reception), 40 cycles compute (2 per recurrent step), 1 cycle C_DONE, 1 pipeline register, 9 cycles send + FSM transitions = 55 total.

## Benchmark Table

| Metric | SW Baseline | HW Accelerator | Speedup |
|--------|-------------|----------------|---------|
| Epoch time | 85.12 s | 24.90 s | 3.42x |
| Step time | 3,405 ms | 996 ms | 3.42x |
| Training throughput | 18.8 samples/sec | 64.27 samples/sec | 3.42x |
| Kernel time/step (all 4 layers) | 2,598 ms | 189.0 ms | 13.7x |
| Kernel calls/sec (Layer 0, 64ch) | 1.97 calls/sec | 1,818,182 calls/sec | 923,636x |
| Layer 0 cycles/call | N/A (software) | 55 | N/A |
| Peak RSS | 2,980 MB | 2,980 MB (host unchanged) | 1.0x |
| Active compute power | ~45 W | 55.80 mW | N/A |
| Energy per sample | 2.3936 J | 0.8682 mJ | 2,757x |

## Speedup Derivation

Kernel time per step (HW): sum of SP_i x cycles_i x 10 ns across all 4 layers = 144.2 + 36.0 + 7.4 + 1.4 = 189.0 ms.

SW non-kernel time per step: 3,405 ms x (1 - 0.763) = 807 ms. This portion runs on the host CPU unchanged.

HW total step time: 189.0 + 807 = 996 ms.

End-to-end speedup: 3,405 / 996 = 3.42x.

Amdahl check: 1 / (0.237 + 0.763/13.7) = 3.42x. Consistent.

## Cycle Count Source

All cycles/call values come from Icarus Verilog simulation of the full DUT (top.sv, interface_mod.sv, compute_core.sv) at 100 MHz with the AXI-S protocol exercised end-to-end. The simulation was run 100 calls with back-to-back transactions. Min, max, and mean cycles are all 55 for T=20 (zero variance). Simulations for T=15 and T=10 confirmed 45 and 35 cycles respectively.

## Energy Efficiency

SW: 45 W / 18.8 samples/sec = 2.3936 J/sample.

HW chiplet: 55.80 mW (OpenROAD post-route, nom_tt_025C_1v80) / 64.27 samples/sec = 0.8682 mJ/sample.

Energy improvement: 2,757x. Note: the host CPU still runs the decoder and optimizer; the chiplet power figure covers the offloaded reservoir kernel only.
