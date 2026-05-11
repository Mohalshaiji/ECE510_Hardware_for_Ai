// =============================================================================
//  codefest/cf06/hdl/crossbar_tb.sv
//
//  Directed + coverage testbench for bw_crossbar_mac
//  Compatible with Icarus Verilog (iverilog -g2012)
//
//  PRIMARY TEST (CF-06 spec)
//  -------------------------
//  Weight matrix W[row][col]  (row = input index i, col = output index j):
//
//    W = | +1  -1  +1  -1 |   row 0
//        | +1  +1  -1  -1 |   row 1
//        | -1  +1  +1  -1 |   row 2
//        | -1  -1  -1  +1 |   row 3
//
//  Input vector: in = [10, 20, 30, 40]
//
//  Hand-calculated expected outputs  out[j] = Sigma_i W[i][j] x in[i]
//  ---------------------------------------------------------------
//  out[0] = (+1x10) + (+1x20) + (-1x30) + (-1x40) =  10+20-30-40 = -40
//  out[1] = (-1x10) + (+1x20) + (+1x30) + (-1x40) = -10+20+30-40 =   0
//  out[2] = (+1x10) + (-1x20) + (+1x30) + (-1x40) =  10-20+30-40 = -20
//  out[3] = (-1x10) + (-1x20) + (-1x30) + (+1x40) = -10-20-30+40 = -20
//
//  Coverage plan (7 test cases)
//  ----------------------------
//  TC_CF06   CF-06 mandatory test (above)
//  TC_ALL_P  All weights +1  -- verifies pure addition path
//  TC_ALL_N  All weights -1  -- verifies pure negation path
//  TC_CHESS  Checkerboard    -- verifies column interleaving
//  TC_MULTI  3-cycle accumulation with CF-06 weights
//  TC_CLR    accum_clr mid-stream
//  TC_BOUND  Boundary inputs +127/+127/-128/-128
// =============================================================================

`timescale 1ns/1ps

module crossbar_tb;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam integer N       = 4;
    localparam integer IN_W    = 8;
    localparam integer ACCUM_W = 20;
    localparam real    TCLK    = 10.0;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    reg                       clk;
    reg                       rst_n;
    reg                       en;
    reg                       weight_we;
    reg  [1:0]                weight_row;
    reg  [1:0]                weight_col;
    reg                       weight_val;
    reg  signed [IN_W-1:0]    data_in   [0:N-1];
    wire signed [ACCUM_W-1:0] accum_out [0:N-1];
    reg                       accum_clr;

    // -----------------------------------------------------------------------
    // Module-level scratch variables (iverilog: no 'automatic' in initial)
    // -----------------------------------------------------------------------
    reg  [N-1:0]              tv_w   [0:N-1];   // weight rows for current TC
    reg  signed [IN_W-1:0]    tv_inp [0:N-1];   // inputs for current TC
    reg  signed [ACCUM_W-1:0] tv_exp [0:N-1];   // expected outputs

    integer pass_cnt;
    integer fail_cnt;
    integer j;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    crossbar_mac #(
        .N      (N),
        .IN_W   (IN_W),
        .ACCUM_W(ACCUM_W)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (en),
        .weight_we  (weight_we),
        .weight_row (weight_row),
        .weight_col (weight_col),
        .weight_val (weight_val),
        .data_in    (data_in),
        .accum_out  (accum_out),
        .accum_clr  (accum_clr)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    initial clk = 1'b0;
    always  #(TCLK/2.0) clk = ~clk;

    // -----------------------------------------------------------------------
    // Task: program full 4x4 weight array from tv_w[]
    //   tv_w[r][c] = 1 => weight +1,  0 => weight -1
    // -----------------------------------------------------------------------
    task load_weights;
        integer r, c;
        begin
            for (r = 0; r < N; r = r + 1) begin
                for (c = 0; c < N; c = c + 1) begin
                    @(negedge clk);
                    weight_we  = 1'b1;
                    weight_row = r[1:0];
                    weight_col = c[1:0];
                    weight_val = tv_w[r][c];
                end
            end
            @(negedge clk);
            weight_we = 1'b0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: apply tv_inp[], pulse en one cycle, deassert
    // -----------------------------------------------------------------------
    task compute_one_cycle;
        integer i;
        begin
            @(negedge clk);
            for (i = 0; i < N; i = i + 1)
                data_in[i] = tv_inp[i];
            en = 1'b1;
            @(posedge clk); #1;
            en = 1'b0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: synchronous accumulator clear
    // -----------------------------------------------------------------------
    task clear_accum;
        begin
            @(negedge clk); accum_clr = 1'b1;
            @(posedge clk); #1;
            accum_clr = 1'b0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: compare accum_out[] against tv_exp[], update scoreboard
    // -----------------------------------------------------------------------
    task check_outputs;
        input [8*16-1:0] tc_label;
        integer          all_ok;
        begin
            all_ok = 1;
            for (j = 0; j < N; j = j + 1) begin
                if (accum_out[j] !== tv_exp[j]) begin
                    all_ok = 0;
                    $display("  [FAIL] %s  out[%0d] = %0d  expected %0d",
                             tc_label, j,
                             $signed(accum_out[j]), $signed(tv_exp[j]));
                    fail_cnt = fail_cnt + 1;
                end else begin
                    $display("  [PASS] %s  out[%0d] = %0d",
                             tc_label, j, $signed(accum_out[j]));
                    pass_cnt = pass_cnt + 1;
                end
            end
            if (all_ok) $display("         -> all outputs match");
        end
    endtask

    // -----------------------------------------------------------------------
    // MAIN TEST SEQUENCE
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);

        // Defaults
        rst_n     = 1'b0;
        en        = 1'b0;
        weight_we = 1'b0;
        weight_row= 2'd0;
        weight_col= 2'd0;
        weight_val= 1'b0;
        accum_clr = 1'b0;
        pass_cnt  = 0;
        fail_cnt  = 0;
        data_in[0] = 0; data_in[1] = 0; data_in[2] = 0; data_in[3] = 0;

        // Reset for 3 cycles
        repeat(3) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        @(negedge clk);

        $display("");
        $display("=== CF-06  Binary-Weight Crossbar MAC Testbench ===");
        $display("");

        // ================================================================
        // TC_CF06 -- mandatory CF-06 test
        //
        //  Weights (bit[c]=1 => +1, 0 => -1):
        //   row0: +1,-1,+1,-1  => 4'b0101
        //   row1: +1,+1,-1,-1  => 4'b0011
        //   row2: -1,+1,+1,-1  => 4'b0110
        //   row3: -1,-1,-1,+1  => 4'b1000
        //
        //  Hand calculation:
        //   out[0]=(+1x10)+(+1x20)+(-1x30)+(-1x40)= 10+20-30-40 = -40
        //   out[1]=(-1x10)+(+1x20)+(+1x30)+(-1x40)=-10+20+30-40 =   0
        //   out[2]=(+1x10)+(-1x20)+(+1x30)+(-1x40)= 10-20+30-40 = -20
        //   out[3]=(-1x10)+(-1x20)+(-1x30)+(+1x40)=-10-20-30+40 = -20
        // ================================================================
        $display("-- TC_CF06: CF-06 specified weight matrix ----------");
        $display("   Weights: row0=[+1,-1,+1,-1]  row1=[+1,+1,-1,-1]");
        $display("            row2=[-1,+1,+1,-1]  row3=[-1,-1,-1,+1]");
        $display("   Input  : [10, 20, 30, 40]");
        $display("   Expected: out[0]=-40  out[1]=0  out[2]=-20  out[3]=-20");

        tv_w[0]=4'b0101; tv_w[1]=4'b0011; tv_w[2]=4'b0110; tv_w[3]=4'b1000;
        tv_inp[0]=8'sd10;  tv_inp[1]=8'sd20; tv_inp[2]=8'sd30; tv_inp[3]=8'sd40;
        tv_exp[0]=-20'sd40; tv_exp[1]=20'sd0; tv_exp[2]=-20'sd20; tv_exp[3]=-20'sd20;
        load_weights; compute_one_cycle; check_outputs("TC_CF06"); clear_accum;
        $display("");

        // ================================================================
        // TC_ALL_P -- all +1 weights, inputs [10,20,30,40]
        //   out[j] = 10+20+30+40 = 100  for all j
        // ================================================================
        $display("-- TC_ALL_P: All +1 weights ------------------------");
        tv_w[0]=4'b1111; tv_w[1]=4'b1111; tv_w[2]=4'b1111; tv_w[3]=4'b1111;
        tv_inp[0]=8'sd10; tv_inp[1]=8'sd20; tv_inp[2]=8'sd30; tv_inp[3]=8'sd40;
        tv_exp[0]=20'sd100; tv_exp[1]=20'sd100; tv_exp[2]=20'sd100; tv_exp[3]=20'sd100;
        load_weights; compute_one_cycle; check_outputs("TC_ALL_P"); clear_accum;
        $display("");

        // ================================================================
        // TC_ALL_N -- all -1 weights, inputs [10,20,30,40]
        //   out[j] = -(10+20+30+40) = -100  for all j
        // ================================================================
        $display("-- TC_ALL_N: All -1 weights ------------------------");
        tv_w[0]=4'b0000; tv_w[1]=4'b0000; tv_w[2]=4'b0000; tv_w[3]=4'b0000;
        tv_inp[0]=8'sd10; tv_inp[1]=8'sd20; tv_inp[2]=8'sd30; tv_inp[3]=8'sd40;
        tv_exp[0]=-20'sd100; tv_exp[1]=-20'sd100; tv_exp[2]=-20'sd100; tv_exp[3]=-20'sd100;
        load_weights; compute_one_cycle; check_outputs("TC_ALL_N"); clear_accum;
        $display("");

        // ================================================================
        // TC_CHESS -- checkerboard weights, inputs [1,2,3,4]
        //   row0=[+1,-1,+1,-1]  row1=[-1,+1,-1,+1]
        //   row2=[+1,-1,+1,-1]  row3=[-1,+1,-1,+1]
        //   out[0]=  1-2+3-4 = -2
        //   out[1]= -1+2-3+4 = +2
        //   out[2]=  1-2+3-4 = -2
        //   out[3]= -1+2-3+4 = +2
        // ================================================================
        $display("-- TC_CHESS: Checkerboard weights ------------------");
        tv_w[0]=4'b0101; tv_w[1]=4'b1010; tv_w[2]=4'b0101; tv_w[3]=4'b1010;
        tv_inp[0]=8'sd1; tv_inp[1]=8'sd2; tv_inp[2]=8'sd3; tv_inp[3]=8'sd4;
        tv_exp[0]=-20'sd2; tv_exp[1]=20'sd2; tv_exp[2]=-20'sd2; tv_exp[3]=20'sd2;
        load_weights; compute_one_cycle; check_outputs("TC_CHESS"); clear_accum;
        $display("");

        // ================================================================
        // TC_MULTI -- 3-cycle accumulation, CF-06 weights
        //   single cycle: [-40, 0, -20, -20]
        //   after 3 cycles: [-120, 0, -60, -60]
        // ================================================================
        $display("-- TC_MULTI: 3-cycle accumulation ------------------");
        tv_w[0]=4'b0101; tv_w[1]=4'b0011; tv_w[2]=4'b0110; tv_w[3]=4'b1000;
        tv_inp[0]=8'sd10; tv_inp[1]=8'sd20; tv_inp[2]=8'sd30; tv_inp[3]=8'sd40;
        tv_exp[0]=-20'sd120; tv_exp[1]=20'sd0; tv_exp[2]=-20'sd60; tv_exp[3]=-20'sd60;
        load_weights;
        compute_one_cycle; compute_one_cycle; compute_one_cycle;
        check_outputs("TC_MULTI"); clear_accum;
        $display("");

        // ================================================================
        // TC_CLR -- 2 cycles, mid-stream clear, 1 more cycle
        //   result after clear+1 cycle = [-40, 0, -20, -20]
        // ================================================================
        $display("-- TC_CLR: accum_clr mid-stream --------------------");
        tv_w[0]=4'b0101; tv_w[1]=4'b0011; tv_w[2]=4'b0110; tv_w[3]=4'b1000;
        tv_inp[0]=8'sd10; tv_inp[1]=8'sd20; tv_inp[2]=8'sd30; tv_inp[3]=8'sd40;
        tv_exp[0]=-20'sd40; tv_exp[1]=20'sd0; tv_exp[2]=-20'sd20; tv_exp[3]=-20'sd20;
        load_weights;
        compute_one_cycle; compute_one_cycle;   // accumulate
        clear_accum;                             // <-- mid-stream clear
        compute_one_cycle;                       // fresh start
        check_outputs("TC_CLR"); clear_accum;
        $display("");

        // ================================================================
        // TC_BOUND -- boundary inputs, all +1 weights
        //   inputs = [127, 127, -128, -128]
        //   out[j] = 127+127-128-128 = -2  for all j
        // ================================================================
        $display("-- TC_BOUND: Boundary inputs (+127,+127,-128,-128) -");
        tv_w[0]=4'b1111; tv_w[1]=4'b1111; tv_w[2]=4'b1111; tv_w[3]=4'b1111;
        tv_inp[0]=8'sd127; tv_inp[1]=8'sd127; tv_inp[2]=-8'sd128; tv_inp[3]=-8'sd128;
        tv_exp[0]=-20'sd2; tv_exp[1]=-20'sd2; tv_exp[2]=-20'sd2; tv_exp[3]=-20'sd2;
        load_weights; compute_one_cycle; check_outputs("TC_BOUND"); clear_accum;
        $display("");

        // ================================================================
        // Summary
        // ================================================================
        $display("=== RESULTS: %0d passed   %0d failed   %0d total ===",
                 pass_cnt, fail_cnt, pass_cnt + fail_cnt);
        if (fail_cnt == 0)
            $display("=== STATUS : ALL TESTS PASSED ===");
        else
            $display("=== STATUS : FAILURES DETECTED -- see log above ===");
        $display("");

        $finish;
    end

    // -----------------------------------------------------------------------
    // Watchdog
    // -----------------------------------------------------------------------
    initial begin
        #100_000;
        $display("WATCHDOG: simulation timeout");
        $finish;
    end

endmodule
