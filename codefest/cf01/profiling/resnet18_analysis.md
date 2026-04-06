# ResNet-18 Profiling Analysis

## Top 5 Layers by MAC Count

| Layer Name | MACs | Parameter Count |
| :--- | :--- | :--- |
| Conv2d: 1-1 | 118,013,952 | 9,408 |
| Conv2d: 3-1 | 115,605,504 | 36,864 |
| Conv2d: 3-4 | 115,605,504 | 36,864 |
| Conv2d: 3-7 | 115,605,504 | 36,864 |
| Conv2d: 3-10 | 115,605,504 | 36,864 |

Note: Where multiple layers had the same MAC count, the first ones encountered were noted.
## Arithmetic Intensity Calculation
**Most MAC-intensive layer:** Conv2d: 1-1

**1. Total Operations (FLOPs):**
* MAC Operations = 118,013,952
* Total FLOPs = 2 * 118,013,952 = 236,027,904 FLOPs

**2. Memory Traffic (Assuming Cold-Miss Only / "No Reuse" from DRAM twice):**
This calculates the strict footprint of bringing the necessary data from DRAM into the processor exactly once.

* **Input Activations Size:** 1 * 3 * 224 * 224 = 150,528 elements
* **Output Activations Size:** 1 * 64 * 112 * 112 = 802,816 elements
* **Total Activations Elements:** 150,528 + 802,816 = 953,344
* **Activations Memory:** 953,344 * 4 Bytes/element = 3,813,376 Bytes

* **Parameters (Weights) Size:** 9,408 elements
* **Weights Memory:** 9,408 * 4 Bytes/element = 37,632 Bytes

* **Total DRAM Traffic:** 3,813,376 Bytes + 37,632 Bytes = 3,851,008 Bytes

**3. Arithmetic Intensity:**
* Total FLOPs / Total DRAM Traffic
* 236,027,904 FLOPs / 3,851,008 Bytes
* = **61.29 FLOPs/Byte**

***

### *Note: Strict Literal Interpretation of "No Reuse"*
The primary calculation above models standard architectural assumptions (a "Cold-Miss Only" cache model). A completely literal, zero-cache interpretation of the phrase "assuming all weights and activations are loaded from DRAM with no reuse" changes the memory footprint significantly. 

Under a strict zero-cache execution, every individual MAC operation independently fetches its required data directly from DRAM:
* **Memory Traffic per MAC:** 1 weight (4 Bytes) + 1 input activation (4 Bytes) + 1 output activation (4 Bytes) = **12 Bytes / MAC**
* **Total Memory Footprint:** 118,013,952 MACs * 12 Bytes = **1,416,167,424 Bytes**
* **Strict Arithmetic Intensity:** 236,027,904 FLOPs / 1,416,167,424 Bytes = **0.166 FLOPs/Byte**
