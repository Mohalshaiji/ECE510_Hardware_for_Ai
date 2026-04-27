`timescale 1ns/1ps

module mac_tb;

    // ── DUT signals ───────────────────────────────────────────────────────
    logic              clk;
    logic              rst;
    logic signed [7:0] a;
    logic signed [7:0] b;
    logic signed [31:0] out;

    // ── instantiate DUT ───────────────────────────────────────────────────
    mac dut (
        .clk(clk),
        .rst(rst),
        .out(out),
        .a(a),
        .b(b)
    );

    // ── 10ns clock ────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── main test sequence ────────────────────────────────────────────────
    integer errors = 0;

    task check(
        input integer expected,
        input string  label
    );
        if (out !== expected) begin
            $display("FAIL [%s] expected=%0d  got=%0d", label, expected, out);
            errors = errors + 1;
        end else begin
            $display("PASS [%s] out=%0d", label, out);
        end
    endtask

    initial begin
        // ── initial reset pulse to bring out out of X ─────────────────────
        rst = 1;
        a   = 0;
        b   = 0;
        @(posedge clk); #1;   // out becomes 0

        rst = 0;

        // ── Phase 1: a=3, b=4 for 3 cycles ───────────────────────────────
        a = 3;
        b = 4;

        @(posedge clk); #1;
        check(12,  "Phase1 cycle1");   // 0 + 12

        @(posedge clk); #1;
        check(24,  "Phase1 cycle2");   // 12 + 12

        @(posedge clk); #1;
        check(36,  "Phase1 cycle3");   // 24 + 12

        // ── Phase 2: assert rst for 1 cycle ───────────────────────────────
        rst = 1;
        a   = 0;
        b   = 0;

        @(posedge clk); #1;
        check(0,   "Reset cycle");

        rst = 0;

        // ── Phase 3: a=-5, b=2 for 2 cycles ──────────────────────────────
        a = -5;
        b =  2;

        @(posedge clk); #1;
        check(-10, "Phase3 cycle1");   // 0 + (-10)

        @(posedge clk); #1;
        check(-20, "Phase3 cycle2");   // -10 + (-10)

        // ── summary ───────────────────────────────────────────────────────
        if (errors == 0)
            $display("\nAll tests PASSED.");
        else
            $display("\n%0d test(s) FAILED.", errors);

        $finish;
    end

endmodule
