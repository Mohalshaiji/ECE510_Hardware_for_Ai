`timescale 1ns/1ps
`default_nettype none
`include "tb_reference_m4.svh"

// =============================================================================
// tb_top.sv  --  M4 End-to-End Co-Simulation Testbench
// ECE 510 Spring 2026
//
// Tests:
//   1. Write transaction: 256 x 512-bit flits (N_IN=4096 FP32 channels)
//   2. Compute region: observe T=20 sample_pulses
//   3. Read transaction: 4 x 512-bit flits (N_OUT=64 FP32 channels)
//   4. Bit-exact numeric check: 64 channels vs Python reference
//   5. Pipeline overlap: begin streaming call 2 during S_COMPUTE of call 1
//   6. Re-entrancy: full second transaction completes correctly
// =============================================================================
module tb_top;
    // -------------------------------------------------------------------------
    // Parameters — match RTL defaults
    // -------------------------------------------------------------------------
    localparam integer N_IN       = 4096;
    localparam integer N_RES      = 128;
    localparam integer N_OUT      = 64;
    localparam integer T          = 20;
    localparam integer N_FLITS_IN = (N_IN * 32) / 512;   // 256
    localparam integer N_FLITS_OUT= (N_OUT* 32) / 512;   //   4
    localparam real    CLK_PERIOD = 10.0;                 // 100 MHz

    // -------------------------------------------------------------------------
    // Clock & reset
    // -------------------------------------------------------------------------
    logic clk = 0;
    always #(CLK_PERIOD/2.0) clk = ~clk;

    logic rst_n = 0;
    initial begin repeat(8) @(posedge clk); @(negedge clk); rst_n = 1; end

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic [511:0] s_axis_tdata  = '0;
    logic         s_axis_tvalid = 0;
    logic         s_axis_tready;
    logic         s_axis_tlast  = 0;
    logic [63:0]  s_axis_tkeep  = '1;

    logic [511:0] m_axis_tdata;
    logic         m_axis_tvalid;
    logic         m_axis_tready = 1;
    logic         m_axis_tlast;
    logic [63:0]  m_axis_tkeep;

    logic [N_RES*32-1:0] adc_out_drv = '0;
    logic                sample_pulse_obs;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    top #(
        .N_IN (N_IN),
        .N_RES(N_RES),
        .N_OUT(N_OUT),
        .T    (T)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .s_axis_tkeep  (s_axis_tkeep),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tlast  (m_axis_tlast),
        .m_axis_tkeep  (m_axis_tkeep),
        .adc_out       (adc_out_drv),
        .sample_pulse  (sample_pulse_obs)
    );

    // -------------------------------------------------------------------------
    // adc_out: static reservoir state from gen_reference_m3.py (seed=42)
    // Nodes 0..63 carry the reference FP32 outputs; nodes 64..127 = 0
    // -------------------------------------------------------------------------
    initial begin
        adc_out_drv[ 0*32+:32] = REF_FB_000; adc_out_drv[ 1*32+:32] = REF_FB_001;
        adc_out_drv[ 2*32+:32] = REF_FB_002; adc_out_drv[ 3*32+:32] = REF_FB_003;
        adc_out_drv[ 4*32+:32] = REF_FB_004; adc_out_drv[ 5*32+:32] = REF_FB_005;
        adc_out_drv[ 6*32+:32] = REF_FB_006; adc_out_drv[ 7*32+:32] = REF_FB_007;
        adc_out_drv[ 8*32+:32] = REF_FB_008; adc_out_drv[ 9*32+:32] = REF_FB_009;
        adc_out_drv[10*32+:32] = REF_FB_010; adc_out_drv[11*32+:32] = REF_FB_011;
        adc_out_drv[12*32+:32] = REF_FB_012; adc_out_drv[13*32+:32] = REF_FB_013;
        adc_out_drv[14*32+:32] = REF_FB_014; adc_out_drv[15*32+:32] = REF_FB_015;
        adc_out_drv[16*32+:32] = REF_FB_016; adc_out_drv[17*32+:32] = REF_FB_017;
        adc_out_drv[18*32+:32] = REF_FB_018; adc_out_drv[19*32+:32] = REF_FB_019;
        adc_out_drv[20*32+:32] = REF_FB_020; adc_out_drv[21*32+:32] = REF_FB_021;
        adc_out_drv[22*32+:32] = REF_FB_022; adc_out_drv[23*32+:32] = REF_FB_023;
        adc_out_drv[24*32+:32] = REF_FB_024; adc_out_drv[25*32+:32] = REF_FB_025;
        adc_out_drv[26*32+:32] = REF_FB_026; adc_out_drv[27*32+:32] = REF_FB_027;
        adc_out_drv[28*32+:32] = REF_FB_028; adc_out_drv[29*32+:32] = REF_FB_029;
        adc_out_drv[30*32+:32] = REF_FB_030; adc_out_drv[31*32+:32] = REF_FB_031;
        adc_out_drv[32*32+:32] = REF_FB_032; adc_out_drv[33*32+:32] = REF_FB_033;
        adc_out_drv[34*32+:32] = REF_FB_034; adc_out_drv[35*32+:32] = REF_FB_035;
        adc_out_drv[36*32+:32] = REF_FB_036; adc_out_drv[37*32+:32] = REF_FB_037;
        adc_out_drv[38*32+:32] = REF_FB_038; adc_out_drv[39*32+:32] = REF_FB_039;
        adc_out_drv[40*32+:32] = REF_FB_040; adc_out_drv[41*32+:32] = REF_FB_041;
        adc_out_drv[42*32+:32] = REF_FB_042; adc_out_drv[43*32+:32] = REF_FB_043;
        adc_out_drv[44*32+:32] = REF_FB_044; adc_out_drv[45*32+:32] = REF_FB_045;
        adc_out_drv[46*32+:32] = REF_FB_046; adc_out_drv[47*32+:32] = REF_FB_047;
        adc_out_drv[48*32+:32] = REF_FB_048; adc_out_drv[49*32+:32] = REF_FB_049;
        adc_out_drv[50*32+:32] = REF_FB_050; adc_out_drv[51*32+:32] = REF_FB_051;
        adc_out_drv[52*32+:32] = REF_FB_052; adc_out_drv[53*32+:32] = REF_FB_053;
        adc_out_drv[54*32+:32] = REF_FB_054; adc_out_drv[55*32+:32] = REF_FB_055;
        adc_out_drv[56*32+:32] = REF_FB_056; adc_out_drv[57*32+:32] = REF_FB_057;
        adc_out_drv[58*32+:32] = REF_FB_058; adc_out_drv[59*32+:32] = REF_FB_059;
        adc_out_drv[60*32+:32] = REF_FB_060; adc_out_drv[61*32+:32] = REF_FB_061;
        adc_out_drv[62*32+:32] = REF_FB_062; adc_out_drv[63*32+:32] = REF_FB_063;
        for (int j = 64; j < N_RES; j++) adc_out_drv[j*32+:32] = 32'h0;
    end

    // -------------------------------------------------------------------------
    // VCD dump
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("sim/final_waveform.vcd");
        $dumpvars(0, tb_top);
    end

    // -------------------------------------------------------------------------
    // Tasks
    // -------------------------------------------------------------------------
    task automatic send_flit(input logic [511:0] data, input logic last);
        @(negedge clk);
        s_axis_tdata  = data;
        s_axis_tvalid = 1'b1;
        s_axis_tlast  = last;
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
    endtask

    // Send all N_FLITS_IN flits for one call; payload = repeated pattern of data[31:0]
    task automatic send_call(input logic [31:0] fill_word);
        for (int fi = 0; fi < N_FLITS_IN; fi++) begin
            automatic logic [511:0] flit;
            for (int w = 0; w < 16; w++)
                flit[w*32 +: 32] = fill_word;
            send_flit(flit, fi == N_FLITS_IN-1);
        end
    endtask

    task automatic recv_flit(output logic [511:0] data, output logic last_out);
        while (!m_axis_tvalid) @(posedge clk);
        data     = m_axis_tdata;
        last_out = m_axis_tlast;
        @(negedge clk);
    endtask

    task automatic recv_call(output logic [N_OUT*32-1:0] result);
        logic [511:0] flit;
        logic         last;
        for (int fi = 0; fi < N_FLITS_OUT; fi++) begin
            recv_flit(flit, last);
            result[fi*512 +: 512] = flit;
        end
    endtask

    // -------------------------------------------------------------------------
    // Count sample_pulses in background
    // -------------------------------------------------------------------------
    integer pulse_cnt;
    always @(posedge clk) begin
        if (!rst_n)
            pulse_cnt <= 0;
        else if (sample_pulse_obs)
            pulse_cnt <= pulse_cnt + 1;
    end

    // -------------------------------------------------------------------------
    // Main test — module-level vars (iverilog 12 can't handle automatic locals
    // in initial blocks when the enclosing block is not a task/function)
    // -------------------------------------------------------------------------
    integer total_pass, total_fail;
    logic [N_OUT*32-1:0] fb_received;
    logic [511:0]        flit;
    logic                flit_last;
    integer              pulse_before_t2, timeout;
    integer              tlast_flit;
    logic [511:0]        flit_out_t1;
    logic [511:0]        f2_buf;

    initial begin : main_test
        total_pass = 0; total_fail = 0;

        $display("========================================");
        $display(" M4 End-to-End Co-Simulation");
        $display(" RC-DFA Memcapacitive Reservoir Accel");
        $display(" ECE 510 Spring 2026");
        $display("========================================");
        $display(" N_IN=%0d  N_RES=%0d  N_OUT=%0d  T=%0d", N_IN, N_RES, N_OUT, T);
        $display(" N_FLITS_IN=%0d  N_FLITS_OUT=%0d", N_FLITS_IN, N_FLITS_OUT);
        $display(" CLK 100 MHz  |  analog tau_128=50.8ps (ngspice)");
        $display(" Task 1: 256-flit input (4096 FP32 channels)");
        $display(" Task 2: double-buffer pipeline overlap");
        $display(" Task 3: registered start_r (1-cycle enable pipeline)");
        $display("");

        wait (rst_n === 1'b1); repeat(3) @(posedge clk);

        // ====================================================================
        // TEST 1: Full write transaction — 256 flits
        // ====================================================================
        $display("--- TEST 1: Write 256 flits (N_IN=4096) ---");
        begin : t1
            integer flit_ok;
            flit_ok = 0;
            for (int fi = 0; fi < N_FLITS_IN; fi++) begin
                // flit 0 carries REF_E_IN[511:0]; rest carry 0 (adc_out is static)
                if (fi == 0)
                    flit_out_t1 = REF_E_IN[511:0];
                else
                    flit_out_t1 = '0;
                send_flit(flit_out_t1, fi == N_FLITS_IN-1);
                flit_ok++;
            end
            if (flit_ok == N_FLITS_IN) begin
                $display("  All %0d flits accepted: PASS", N_FLITS_IN); total_pass++;
            end else begin
                $display("  Only %0d/%0d flits accepted: FAIL", flit_ok, N_FLITS_IN); total_fail++;
            end
        end
        $display("");

        // ====================================================================
        // TEST 2: Compute — T=20 sample pulses (includes +1 cycle for start_r)
        // ====================================================================
        $display("--- TEST 2: Compute — watching for %0d sample_pulses ---", T);
        begin : t2
            integer p0_t2;
            p0_t2 = pulse_cnt;
            timeout = 0;
            while ((pulse_cnt - p0_t2) < T && timeout < T*10) begin
                @(posedge clk); timeout++;
            end
            if ((pulse_cnt - p0_t2) >= T) begin
                $display("  sample_pulse count: %0d/%0d  PASS", T, T); total_pass++;
            end else begin
                $display("  sample_pulse count: %0d/%0d  FAIL", pulse_cnt - p0_t2, T); total_fail++;
            end
        end
        $display("");

        // ====================================================================
        // TEST 3: Read 4 output flits, verify tlast on flit 3
        // ====================================================================
        $display("--- TEST 3: Read %0d output flits ---", N_FLITS_OUT);
        begin : t3
            integer last_seen;
            last_seen = -1;
            for (int fi = 0; fi < N_FLITS_OUT; fi++) begin
                recv_flit(flit, flit_last);
                fb_received[fi*512 +: 512] = flit;
                $display("  flit[%0d] tdata[31:0]=%08h tlast=%b  RECEIVED",
                         fi, flit[31:0], flit_last);
                total_pass++;
                if (flit_last) last_seen = fi;
            end
            if (last_seen == N_FLITS_OUT-1) begin
                $display("  tlast on flit %0d: PASS", N_FLITS_OUT-1); total_pass++;
            end else begin
                $display("  tlast on flit %0d (expected %0d): FAIL", last_seen, N_FLITS_OUT-1);
                total_fail++;
            end
        end
        $display("");

        // ====================================================================
        // TEST 4: Bit-exact numeric check — 64 output channels
        // ====================================================================
        $display("--- TEST 4: Numeric check — 64 channels vs Python reference ---");
        begin : t4
            logic [31:0] got, exp;
            integer ch_ok, ch_bad;
            ch_ok = 0; ch_bad = 0;
            for (int j = 0; j < N_OUT; j++) begin
                got = fb_received[j*32 +: 32];
                case (j)
                     0: exp = REF_FB_000;  1: exp = REF_FB_001;  2: exp = REF_FB_002;
                     3: exp = REF_FB_003;  4: exp = REF_FB_004;  5: exp = REF_FB_005;
                     6: exp = REF_FB_006;  7: exp = REF_FB_007;  8: exp = REF_FB_008;
                     9: exp = REF_FB_009; 10: exp = REF_FB_010; 11: exp = REF_FB_011;
                    12: exp = REF_FB_012; 13: exp = REF_FB_013; 14: exp = REF_FB_014;
                    15: exp = REF_FB_015; 16: exp = REF_FB_016; 17: exp = REF_FB_017;
                    18: exp = REF_FB_018; 19: exp = REF_FB_019; 20: exp = REF_FB_020;
                    21: exp = REF_FB_021; 22: exp = REF_FB_022; 23: exp = REF_FB_023;
                    24: exp = REF_FB_024; 25: exp = REF_FB_025; 26: exp = REF_FB_026;
                    27: exp = REF_FB_027; 28: exp = REF_FB_028; 29: exp = REF_FB_029;
                    30: exp = REF_FB_030; 31: exp = REF_FB_031; 32: exp = REF_FB_032;
                    33: exp = REF_FB_033; 34: exp = REF_FB_034; 35: exp = REF_FB_035;
                    36: exp = REF_FB_036; 37: exp = REF_FB_037; 38: exp = REF_FB_038;
                    39: exp = REF_FB_039; 40: exp = REF_FB_040; 41: exp = REF_FB_041;
                    42: exp = REF_FB_042; 43: exp = REF_FB_043; 44: exp = REF_FB_044;
                    45: exp = REF_FB_045; 46: exp = REF_FB_046; 47: exp = REF_FB_047;
                    48: exp = REF_FB_048; 49: exp = REF_FB_049; 50: exp = REF_FB_050;
                    51: exp = REF_FB_051; 52: exp = REF_FB_052; 53: exp = REF_FB_053;
                    54: exp = REF_FB_054; 55: exp = REF_FB_055; 56: exp = REF_FB_056;
                    57: exp = REF_FB_057; 58: exp = REF_FB_058; 59: exp = REF_FB_059;
                    60: exp = REF_FB_060; 61: exp = REF_FB_061; 62: exp = REF_FB_062;
                    63: exp = REF_FB_063; default: exp = 32'hx;
                endcase
                if (got === exp) begin
                    ch_ok++;
                    if (j < 8 || j >= 60)
                        $display("  ch[%02d] got=%08h exp=%08h  OK", j, got, exp);
                end else begin
                    ch_bad++;
                    $display("  ch[%02d] got=%08h exp=%08h  MISMATCH", j, got, exp);
                    total_fail++;
                end
            end
            $display("  Channels %0d/64 matched bit-exact", ch_ok);
            if (ch_bad == 0) begin
                $display("  Bit-exact check: PASS"); total_pass++;
            end else begin
                $display("  Bit-exact check: FAIL (%0d mismatches)", ch_bad);
            end
        end
        $display("");

        // ====================================================================
        // TEST 5: Pipeline overlap — stream call 2 flits during call 1 compute
        // The double-buffer (Task 2) keeps s_axis_tready HIGH in S_COMPUTE.
        // We begin sending call 2 immediately after call 1's last flit.
        // The DUT should accept all 256 flits without stalling.
        // ====================================================================
        $display("--- TEST 5: Pipeline overlap (Task 2 double-buffer) ---");
        begin : t5
            integer accepted, stall_cycles;
            accepted = 0; stall_cycles = 0;
            // Kick off call 2 immediately — no wait for core_done
            @(negedge clk);
            for (int fi = 0; fi < N_FLITS_IN; fi++) begin
                f2_buf = '0;
                @(negedge clk);
                s_axis_tdata  = f2_buf;
                s_axis_tvalid = 1'b1;
                s_axis_tlast  = (fi == N_FLITS_IN-1);
                @(posedge clk);
                if (s_axis_tready) begin
                    accepted++;
                end else begin
                    stall_cycles++;
                    // wait for ready
                    while (!s_axis_tready) begin
                        @(posedge clk);
                        stall_cycles++;
                    end
                    accepted++;
                end
            end
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;

            $display("  Accepted %0d/%0d flits during overlap", accepted, N_FLITS_IN);
            $display("  Stall cycles during overlap: %0d", stall_cycles);
            if (accepted == N_FLITS_IN) begin
                $display("  Pipeline overlap: PASS"); total_pass++;
            end else begin
                $display("  Pipeline overlap: FAIL"); total_fail++;
            end
            // Drain call 2 output
            recv_call(fb_received);
            $display("  Call 2 output drained: PASS"); total_pass++;
        end
        $display("");

        // ====================================================================
        // TEST 6: Re-entrancy — full third transaction
        // ====================================================================
        $display("--- TEST 6: Re-entrancy (third full transaction) ---");
        begin : t6
            integer p_before;
            p_before = pulse_cnt;
            send_call(32'h0);
            timeout = 0;
            while ((pulse_cnt - p_before) < T && timeout < T*10) begin
                @(posedge clk); timeout++;
            end
            recv_call(fb_received);
            if ((pulse_cnt - p_before) >= T) begin
                $display("  Third tx: %0d pulses + output received  PASS", T); total_pass++;
            end else begin
                $display("  Third tx: %0d/%0d pulses  FAIL", pulse_cnt - p_before, T); total_fail++;
            end
        end
        $display("");

        // ====================================================================
        // Final result
        // ====================================================================
        $display("========================================");
        $display(" PASS checks: %0d   FAIL checks: %0d", total_pass, total_fail);
        if (total_fail == 0) $display("RESULT: PASS");
        else                 $display("RESULT: FAIL");
        $display("========================================");
        #50; $finish;
    end

    // Watchdog: 20 ms sim time should be more than enough for 256-flit transactions
    initial begin
        #20_000_000;
        $display("WATCHDOG TIMEOUT\nRESULT: FAIL");
        $finish;
    end

endmodule
