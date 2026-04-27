# MAC Code Review
**Course:** ECE 410/510 — Hardware for Artificial Intelligence and Machine Learning  
**Assignment:** Codefest 4 — CLLM  
**Date:** 2026-04-26  

---

## 1. LLM Model Versions

| File | LLM Used |
|---|---|
| `mac_llm_A.v` | Claude Sonnet 4.6 (claude-sonnet-4-6) |
| `mac_llm_B.v` | Qwen 2.5 Coder:7b |

---

## 2. Prompt Used (identical for both LLMs)

> Write a SystemVerilog module named `mac` that implements a multiply-accumulate unit.
>
> **1. Module Signature** — Use the exact port names, types, and bit widths below:
> - Inputs:
>   - `clk` (1-bit logic)
>   - `rst` (1-bit logic)
>   - `a` (8-bit signed logic: `logic signed [7:0]`)
>   - `b` (8-bit signed logic: `logic signed [7:0]`)
> - Output:
>   - `out` (32-bit signed logic: `logic signed [31:0]`)
>
> **2. Reset Behavior** — Implement a synchronous, active-high reset using the `rst` signal.
>
> **3. Output Register Behavior** — Use an `always_ff` block triggered on the rising edge of `clk`. Inside the block:
> - If `rst` is high, reset `out` to 0.
> - If `rst` is low, multiply `a` by `b` and add the result to the current value of `out`.
>
> **4. Strict Constraints**
> - Provide synthesizable SystemVerilog only.
> - Do NOT include any `initial` blocks.
> - Do NOT include any `$display` or print statements.
> - Name the file: mac_llm_A.v (or mac_llm_B.v for the second LLM).
> - Do NOT use any artificial delays (e.g., `#` symbols).

---

## 3. Source Listings

### mac_llm_A.v
```systemverilog
module mac (
    input  logic              clk,
    input  logic              rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + (a * b);
    end

endmodule
```

### mac_llm_B.v
```systemverilog
module mac (
    input  logic          clk,
    input  logic          rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 0;
        end else begin
            out <= out + (a * b);
        end
    end
endmodule
```

---

## 4. Compilation

Both files compiled cleanly with:

```powershell
iverilog -g2012 -o mac_sim.vvp mac_llm_A.v mac_tb.v
iverilog -g2012 -o mac_sim.vvp mac_llm_B.v mac_tb.v
```

No errors or warnings for either file.

---

## 5. Simulation Results

Testbench: `mac_tb.v`  
Simulator: Icarus Verilog (iverilog -g2012)

**Sequence applied:**
1. Initial reset pulse (1 cycle) — clears accumulator from X to 0
2. `a=3, b=4` for 3 cycles
3. `rst=1` for 1 cycle
4. `a=-5, b=2` for 2 cycles

### mac_llm_A.v output
```
PASS [Phase1 cycle1] out=12
PASS [Phase1 cycle2] out=24
PASS [Phase1 cycle3] out=36
PASS [Reset cycle]   out=0
PASS [Phase3 cycle1] out=-10
PASS [Phase3 cycle2] out=-20

All tests PASSED.
mac_tb.v:88: $finish called at 66000 (1ps)
```

### mac_llm_B.v output
```
PASS [Phase1 cycle1] out=12
PASS [Phase1 cycle2] out=24
PASS [Phase1 cycle3] out=36
PASS [Reset cycle]   out=0
PASS [Phase3 cycle1] out=-10
PASS [Phase3 cycle2] out=-20

All tests PASSED.
mac_tb.v:88: $finish called at 66000 (1ps)
```

Both DUTs pass all 6 assertions.

---

## 6. Code Review — Issues Found

### Issue 1 — Reset literal type (mac_llm_B.v)

**Offending line:**
```systemverilog
out <= 0;
```

**Explanation:**  
In `mac_llm_B.v` the reset assignment uses a plain unsized integer literal `0`. While most synthesizers
will infer the correct 32-bit zero from context, the SystemVerilog LRM specifies that an unsized literal
defaults to 32 bits of type `integer` (signed). Assigning it to a `logic signed [31:0]` port works here
by coincidence of width, but it is considered ambiguous style — the intent is not self-documenting, and
in a wider or differently-typed accumulator this could silently truncate or sign-extend incorrectly.

**Corrected version:**
```systemverilog
out <= 32'sd0;
```

---

### Issue 2 — Implicit product width and sign extension (both files, latent risk)

**Offending line (both files):**
```systemverilog
out <= out + (a * b);
```

**Explanation:**  
`a` and `b` are both `logic signed [7:0]`. Multiplying two 8-bit signed operands produces a 16-bit
signed result (IEEE 1800-2012 §11.6.1). When this 16-bit product is added to the 32-bit `out`, the
narrower operand is sign-extended implicitly. Both files work correctly in simulation because iverilog
follows the standard, but the implicit promotion is not obvious and may be flagged by stricter lint tools.
The corrected file uses an explicit `assign` for the product and a manual sign-extension concatenation
to make the intent auditable.

Note: the `32'(signed'(...))` cast syntax is valid per the SystemVerilog LRM but is **not supported by
iverilog**. The portable alternative is explicit bit-replication: `{{16{product[15]}}, product}`.

**Corrected version (mac_correct.v):**
```systemverilog
logic signed [15:0] product;
assign product = a * b;

always_ff @(posedge clk) begin
    if (rst) begin
        out <= 32'sd0;
    end else begin
        out <= out + {{16{product[15]}}, product};  // explicit sign-extension
    end
end
```

---

### Issue 3 — Missing `begin`/`end` on single-statement branches (mac_llm_A.v)

**Offending lines:**
```systemverilog
    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + (a * b);
    end
```

**Explanation:**  
The `if` and `else` branches each contain a single statement and omit `begin`/`end`. This is
syntactically legal, but is a well-known source of maintenance bugs — adding a second statement to a
branch without adding `begin`/`end` silently moves it outside the conditional. Most RTL style guides
(e.g., lowRISC, Cliff Cummings) mandate `begin`/`end` on all branches inside `always_ff` blocks.
`mac_llm_B.v` already follows this style correctly.

**Corrected version:**
```systemverilog
always_ff @(posedge clk) begin
    if (rst) begin
        out <= 32'sd0;
    end else begin
        out <= out + (a * b);
    end
end
```

---

## 7. Summary Table

| # | Issue | Affects | Severity | Simulation impact |
|---|---|---|---|---|
| 1 | Unsized reset literal `0` instead of `32'sd0` | B only | Low (style) | None with current width |
| 2 | Implicit signed product width promotion; `32'(signed'())` cast unsupported by iverilog | Both | Medium (portability) | None in iverilog; compile error with cast syntax |
| 3 | Missing `begin`/`end` on single-line branches | A only | Low (maintainability) | None |

Neither file contains non-synthesizable constructs, wrong process types, wrong reset polarity, or
missing port directions. Both are functionally correct for the given specification.

---

## 8. Corrected File

`mac_correct.v` incorporates all three fixes: explicit `32'sd0` reset, a named `product` wire with
manual sign-extension via bit-replication, and `begin`/`end` on all branches. It compiles cleanly with:

```powershell
iverilog -g2012 -o mac_sim.vvp mac_correct.v mac_tb.v
vvp mac_sim.vvp
```

and passes all 6 testbench assertions.
