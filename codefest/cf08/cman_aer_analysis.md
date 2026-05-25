# CMAN AER Bandwidth Analysis
Codefest 8 | ECE 410/510 | Spring 2026

## Task 1: Mean Aggregate Spike Rate

R = N x f = 1024 x 50 = 51,200 spikes/second

## Task 2: Mean Required AER Bandwidth

Each AER packet is 20 bits (10-bit address + 6-bit timestamp + 4-bit framing/parity overhead).

B = R x 20 = 51,200 x 20 = 1,024,000 bits/second = 1.024 Mbit/s

## Task 3: Interface Comparison

| Interface | Max Bandwidth | Sustains 1.024 Mbit/s |
|-----------|---------------|----------------------|
| SPI | 50 Mbit/s | Yes |
| I2C | 3.4 Mbit/s | Yes |
| AXI4-Lite | 100 Mbit/s | Yes |

All three interfaces sustain the mean rate. The lowest-complexity interface that suffices is I2C, since 1.024 Mbit/s is within its 3.4 Mbit/s limit.

## Task 4: Burst Analysis

25% of 1024 neurons firing in a 1 ms window = 256 spikes.

Peak bandwidth = (256 x 20 bits) / 0.001 s = 5,120,000 bits/s = 5.12 Mbit/s

Burst-to-mean ratio = 5.12 / 1.024 = 5

The peak burst of 5.12 Mbit/s exceeds I2C's 3.4 Mbit/s limit, so buffering is required. A buffer of approximately 5 Mbit depth is needed to absorb the full burst. SPI and AXI4-Lite can handle the peak without buffering.

## Task 5: Frame-Based Comparison

Frame-based bandwidth = 1024 bits / 0.001 s = 1,024,000 bits/s = 1.024 Mbit/s

AER bandwidth at f = 50 Hz = 1.024 Mbit/s (from Task 2)

AER-to-frame ratio at f = 50 Hz = 1.024 / 1.024 = 1.0

Crossover firing rate: set N x f x 20 = N x 1000, solve for f = 1000 / 20 = 50 Hz

AER is more bandwidth efficient than frame-based readout when the mean neuron firing rate is below 50 Hz; above 50 Hz, frame-based readout uses less bandwidth.
