\# CF05 CMAN: Systolic Array Trace





\## Task 1: PE Diagram (Table Format)

A $2\\times2$ weight-stationary systolic array computes $C = A \\times B$. The weights from matrix $B$ are pre-loaded into the processing elements (PEs) and remain fixed. Inputs from matrix $A$ stream in from the left (shifting right), while partial sums accumulate downward.



| | \*\*Column 0\*\*<br>\*(Inputs shift right $\\rightarrow$)\* | \*\*Column 1\*\*<br>\*(Inputs shift right $\\rightarrow$)\* |

| :--- | :--- | :--- |

| \*\*Row 0\*\*<br>\*(Sums shift down $\\downarrow$)\* | \*\*PE\[0]\[0]\*\*<br>Weight = 5 | \*\*PE\[0]\[1]\*\*<br>Weight = 6 |

| \*\*Row 1\*\*<br>\*(Sums shift down $\\downarrow$)\* | \*\*PE\[1]\[0]\*\*<br>Weight = 7 | \*\*PE\[1]\[1]\*\*<br>Weight = 8 |



\## Task 2: Cycle Table

Here is the cycle-by-cycle trace for the execution. Inputs from matrix A stream in (A\[0,0]=1, A\[0,1]=2, A\[1,0]=3, A\[1,1]=4), and partial sums accumulate downward. 



| Cycle | Input to row 0 | Input to row 1 | PE\[0]\[0] PS | PE\[0]\[1] PS | PE\[1]\[0] PS | PE\[1]\[1] PS | Output C values |

| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| \*\*0\*\* | A\[0,0] = 1 | - | 5 | - | - | - | - |

| \*\*1\*\* | A\[1,0] = 3 | A\[0,1] = 2 | 15 | 6 | 19 | - | C\[0,0] = 19 |

| \*\*2\*\* | - | A\[1,1] = 4 | - | 18 | 43 | 22 | C\[1,0] = 43, C\[0,1] = 22 |

| \*\*3\*\* | - | - | - | - | - | 50 | C\[1,1] = 50 |

| \*\*4\*\* | - | - | - | - | - | - | (All outputs complete) |



\## Task 3: Stats

Based on the array operations:

\* \*\*(a) Total MAC operations:\*\* 8 MAC operations are performed to multiply two 2x2 matrices.

\* \*\*(b) Input value reuse:\*\* Each input value from matrix A is reused 1 time (it passes through two PEs).

\* \*\*(c) Off-chip memory accesses:\*\*

&#x20;   \* \*\*A (inputs):\*\* 4 memory accesses.

&#x20;   \* \*\*B (weights):\*\* 4 memory accesses.

&#x20;   \* \*\*C (outputs):\*\* 4 memory accesses.



\## Task 4: Output-Stationary

If this were an output-stationary array instead, the partial sums (which become the final values of matrix C) would stay fixed inside the PEs, while the values from both input matrices A and B would stream through the array.

