# Software Baseline Benchmark — RC-DFA VAE
## ECE 510 Spring 2026 — Project Milestone 1

## Platform

| Field | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Python | 3.10.6 |
| PyTorch | 2.12.0.dev20260407+cu128 |
| Device | CPU (forced) |
| CPU threads | 24 |
| Batch size | 64 |
| Latent channels | 16 |
| Patch size | 128 × 128 × 4 (BCHW) |
| Model params | 3,501,476 |

## Execution Time (wall-clock, median over 10 runs)

| Metric | Value |
|---|---|
| Median epoch time | 85.12s |
| Mean epoch time | 86.51s |
| Min epoch time | 84.99s |
| Max epoch time | 89.41s |
| Steps per epoch | 25 (1600 train patches ÷ batch size 64) |
| ms/step | 3405ms |
| Throughput | 18.8 samples/sec |

## Memory Usage

| Metric | Value |
|---|---|
| Peak RSS | 2980 MB |

## Per-Epoch Times (10 epochs, seconds)

| Epoch | Time (s) |
|---|---|
| 1 | 89.41 |
| 2 | 87.13 |
| 3 | 86.44 |
| 4 | 95.74 |
| 5 | 99.08 |
| 6 | 86.07 |
| 7 | 84.61 |
| 8 | 85.03 |
| 9 | 85.54 |
| 10 | 85.42 |

## Notes

- Timing uses `time.time()` per epoch, median over 10 epochs.
- Single CPU execution, no GPU (forced via `torch.device('cpu')`).
- Epochs 4–5 are elevated due to loss curriculum switch at epoch 3 (MSE → MS-SSIM+L1).
- Stable baseline is epochs 6–10: median 85.42s/epoch.
- For M4 speedup comparison: reproduce by running `tiled_profile.py` with `N_EPOCHS=10`
  on the same hardware with `nyu_depth_v2_labeled.mat` present.
