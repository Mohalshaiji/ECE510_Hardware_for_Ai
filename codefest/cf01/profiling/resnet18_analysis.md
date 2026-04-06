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
**Total FLOPs:** 2 * 118,013,952 = 236,027,904

### Method 1: Strict "No Reuse" Assumption (Assignment Interpretation)
This calculation assumes every single MAC operation loads its weights and activations directly from DRAM independently.

* **Memory Traffic per MAC:** 1 weight (4B) + 1 input activation (4B) + 1 output activation (4B) = 12 Bytes/MAC
* **Total Memory Footprint:** 118,013,952 MACs * 12 Bytes = 1,416,167,424 Bytes
* **Arithmetic Intensity:** 236,027,904 FLOPs / 1,416,167,424 Bytes = **0.166 FLOPs/Byte**

### Method 2: Traditional "Weight Reuse" (Real-World Execution)
This calculation models actual hardware behavior, where the small convolutional filter is loaded into cache once and reused across the entire input image volume. 

* **Total Activations:** (150,528 input nodes + 802,816 output nodes) * 4 Bytes = 3,813,376 Bytes
* **Total Weights (Parameters):** 9,408 parameters * 4 Bytes = 37,632 Bytes
* **Total Memory Footprint:** 3,813,376 Bytes + 37,632 Bytes = 3,851,008 Bytes
* **Arithmetic Intensity:** 236,027,904 FLOPs / 3,851,008 Bytes = **61.29 FLOPs/Byte**
