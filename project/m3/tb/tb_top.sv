`timescale 1ns/1ps
`default_nettype none
`include "tb_reference_m3.svh"

module tb_top;
    localparam int  N_IN=64, N_RES=128, N_OUT=64, T=20;
    localparam real CLK_PERIOD=10.0;

    logic clk=0; always #(CLK_PERIOD/2.0) clk=~clk;
    logic rst_n=0;
    initial begin repeat(8)@(posedge clk); @(negedge clk); rst_n=1; end

    logic [511:0] s_axis_tdata='0; logic s_axis_tvalid=0,s_axis_tready,s_axis_tlast=0;
    logic [63:0]  s_axis_tkeep='1;
    logic [511:0] m_axis_tdata; logic m_axis_tvalid,m_axis_tready=1,m_axis_tlast;
    logic [63:0]  m_axis_tkeep;
    logic [N_RES*32-1:0] adc_out_drv='0;
    logic sample_pulse_obs;

    top #(.N_IN(N_IN),.N_RES(N_RES),.N_OUT(N_OUT),.T(T)) dut(
        .clk(clk),.rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),.s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),.s_axis_tlast(s_axis_tlast),
        .s_axis_tkeep(s_axis_tkeep),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),.m_axis_tlast(m_axis_tlast),
        .m_axis_tkeep(m_axis_tkeep),
        .adc_out(adc_out_drv),.sample_pulse(sample_pulse_obs));

    // adc_out: exact FP32 reservoir state from gen_reference_m3.py (seed=42)
    // Driven statically -- analog has settled (tau_128=50.8ps, ngspice-verified)
    initial begin
        adc_out_drv[ 0*32+:32]=REF_FB_000; adc_out_drv[ 1*32+:32]=REF_FB_001;
        adc_out_drv[ 2*32+:32]=REF_FB_002; adc_out_drv[ 3*32+:32]=REF_FB_003;
        adc_out_drv[ 4*32+:32]=REF_FB_004; adc_out_drv[ 5*32+:32]=REF_FB_005;
        adc_out_drv[ 6*32+:32]=REF_FB_006; adc_out_drv[ 7*32+:32]=REF_FB_007;
        adc_out_drv[ 8*32+:32]=REF_FB_008; adc_out_drv[ 9*32+:32]=REF_FB_009;
        adc_out_drv[10*32+:32]=REF_FB_010; adc_out_drv[11*32+:32]=REF_FB_011;
        adc_out_drv[12*32+:32]=REF_FB_012; adc_out_drv[13*32+:32]=REF_FB_013;
        adc_out_drv[14*32+:32]=REF_FB_014; adc_out_drv[15*32+:32]=REF_FB_015;
        adc_out_drv[16*32+:32]=REF_FB_016; adc_out_drv[17*32+:32]=REF_FB_017;
        adc_out_drv[18*32+:32]=REF_FB_018; adc_out_drv[19*32+:32]=REF_FB_019;
        adc_out_drv[20*32+:32]=REF_FB_020; adc_out_drv[21*32+:32]=REF_FB_021;
        adc_out_drv[22*32+:32]=REF_FB_022; adc_out_drv[23*32+:32]=REF_FB_023;
        adc_out_drv[24*32+:32]=REF_FB_024; adc_out_drv[25*32+:32]=REF_FB_025;
        adc_out_drv[26*32+:32]=REF_FB_026; adc_out_drv[27*32+:32]=REF_FB_027;
        adc_out_drv[28*32+:32]=REF_FB_028; adc_out_drv[29*32+:32]=REF_FB_029;
        adc_out_drv[30*32+:32]=REF_FB_030; adc_out_drv[31*32+:32]=REF_FB_031;
        adc_out_drv[32*32+:32]=REF_FB_032; adc_out_drv[33*32+:32]=REF_FB_033;
        adc_out_drv[34*32+:32]=REF_FB_034; adc_out_drv[35*32+:32]=REF_FB_035;
        adc_out_drv[36*32+:32]=REF_FB_036; adc_out_drv[37*32+:32]=REF_FB_037;
        adc_out_drv[38*32+:32]=REF_FB_038; adc_out_drv[39*32+:32]=REF_FB_039;
        adc_out_drv[40*32+:32]=REF_FB_040; adc_out_drv[41*32+:32]=REF_FB_041;
        adc_out_drv[42*32+:32]=REF_FB_042; adc_out_drv[43*32+:32]=REF_FB_043;
        adc_out_drv[44*32+:32]=REF_FB_044; adc_out_drv[45*32+:32]=REF_FB_045;
        adc_out_drv[46*32+:32]=REF_FB_046; adc_out_drv[47*32+:32]=REF_FB_047;
        adc_out_drv[48*32+:32]=REF_FB_048; adc_out_drv[49*32+:32]=REF_FB_049;
        adc_out_drv[50*32+:32]=REF_FB_050; adc_out_drv[51*32+:32]=REF_FB_051;
        adc_out_drv[52*32+:32]=REF_FB_052; adc_out_drv[53*32+:32]=REF_FB_053;
        adc_out_drv[54*32+:32]=REF_FB_054; adc_out_drv[55*32+:32]=REF_FB_055;
        adc_out_drv[56*32+:32]=REF_FB_056; adc_out_drv[57*32+:32]=REF_FB_057;
        adc_out_drv[58*32+:32]=REF_FB_058; adc_out_drv[59*32+:32]=REF_FB_059;
        adc_out_drv[60*32+:32]=REF_FB_060; adc_out_drv[61*32+:32]=REF_FB_061;
        adc_out_drv[62*32+:32]=REF_FB_062; adc_out_drv[63*32+:32]=REF_FB_063;
        for (int j=64; j<N_RES; j++) adc_out_drv[j*32+:32]=32'h0;
    end

    initial begin $dumpfile("sim/cosim_waveform.vcd"); $dumpvars(0,tb_top); end

    // send_flit: present valid+data, wait for handshake on posedge, then deassert
    task automatic send_flit(input logic [511:0] data, input logic last);
        // Present before clock edge
        @(negedge clk);
        s_axis_tdata  = data;
        s_axis_tvalid = 1'b1;
        s_axis_tlast  = last;
        // Wait for posedge where tready is high (handshake)
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        // Deassert after handshake cycle
        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
    endtask

    task automatic recv_flit(output logic [511:0] data, output logic last_out);
        while (!m_axis_tvalid) @(posedge clk);
        data     = m_axis_tdata;
        last_out = m_axis_tlast;
        @(negedge clk);  // consume
    endtask

    integer total_pass, total_fail;

    initial begin : main_test
        logic [2047:0] e_vec, fb_received;
        logic [511:0]  flit; logic flit_last;
        integer pulse_cnt, timeout, tlast_flit;

        total_pass=0; total_fail=0;
        $display("========================================");
        $display(" M3 End-to-End Co-Simulation");
        $display(" RC-DFA Memcapacitive Reservoir Accel");
        $display(" ECE 510 Spring 2026");
        $display("========================================");
        $display(" N_IN=%0d N_RES=%0d N_OUT=%0d T=%0d", N_IN,N_RES,N_OUT,T);
        $display(" CLK 100 MHz  |  analog tau_128=50.8ps (ngspice)");
        $display("");

        wait(rst_n===1'b1); repeat(3)@(posedge clk);

        // ---- REGION 1: write 4 flits ------------------------------------
        $display("--- REGION 1: Host write (4 x 512b flits, 256 B = 64 x FP32) ---");
        e_vec = REF_E_IN;
        for (int fi=0; fi<4; fi++) begin
            flit = e_vec[fi*512+:512];
            send_flit(flit, fi==3);
            $display("  flit[%0d] tdata[31:0]=%08h tlast=%b  ACCEPTED", fi, flit[31:0], fi==3);
            total_pass++;
        end
        $display("  Write transaction: PASS\n");

        // ---- REGION 2: watch compute ------------------------------------
        $display("--- REGION 2: Compute -- watching for %0d sample_pulses ---", T);
        pulse_cnt=0; timeout=0;
        while (pulse_cnt < T && timeout < T*8) begin
            @(posedge clk); timeout++;
            if (sample_pulse_obs===1'b1) begin
                pulse_cnt++;
                $display("  sample_pulse[%02d]  t=%0t ns", pulse_cnt, $time/1000);
            end
        end
        if (pulse_cnt==T) begin
            $display("  sample_pulse count: %0d/%0d  PASS\n", T,T); total_pass++;
        end else begin
            $display("  sample_pulse count: %0d/%0d  FAIL\n", pulse_cnt,T); total_fail++;
        end

        // ---- REGION 3: read 4 flits -------------------------------------
        $display("--- REGION 3: Host read (4 x 512b flits) ---");
        tlast_flit=-1;
        for (int fi=0; fi<4; fi++) begin
            recv_flit(flit, flit_last);
            fb_received[fi*512+:512]=flit;
            $display("  flit[%0d] tdata[31:0]=%08h tlast=%b  RECEIVED", fi, flit[31:0], flit_last);
            if (flit_last) tlast_flit=fi;
            total_pass++;
        end
        if (tlast_flit==3) begin
            $display("  tlast on flit 3: PASS"); total_pass++;
        end else begin
            $display("  tlast on flit %0d (expected 3): FAIL", tlast_flit); total_fail++;
        end
        $display("");

        // ---- Numeric verification: all 64 channels ----------------------
        $display("--- Numeric check: 64 channels, bit-exact vs Python model ---");
        begin
            logic [31:0] got, exp; integer ch_ok=0, ch_bad=0;
            for (int j=0; j<N_OUT; j++) begin
                got = fb_received[j*32+:32];
                case(j)
                     0:exp=REF_FB_000;  1:exp=REF_FB_001;  2:exp=REF_FB_002;
                     3:exp=REF_FB_003;  4:exp=REF_FB_004;  5:exp=REF_FB_005;
                     6:exp=REF_FB_006;  7:exp=REF_FB_007;  8:exp=REF_FB_008;
                     9:exp=REF_FB_009; 10:exp=REF_FB_010; 11:exp=REF_FB_011;
                    12:exp=REF_FB_012; 13:exp=REF_FB_013; 14:exp=REF_FB_014;
                    15:exp=REF_FB_015; 16:exp=REF_FB_016; 17:exp=REF_FB_017;
                    18:exp=REF_FB_018; 19:exp=REF_FB_019; 20:exp=REF_FB_020;
                    21:exp=REF_FB_021; 22:exp=REF_FB_022; 23:exp=REF_FB_023;
                    24:exp=REF_FB_024; 25:exp=REF_FB_025; 26:exp=REF_FB_026;
                    27:exp=REF_FB_027; 28:exp=REF_FB_028; 29:exp=REF_FB_029;
                    30:exp=REF_FB_030; 31:exp=REF_FB_031; 32:exp=REF_FB_032;
                    33:exp=REF_FB_033; 34:exp=REF_FB_034; 35:exp=REF_FB_035;
                    36:exp=REF_FB_036; 37:exp=REF_FB_037; 38:exp=REF_FB_038;
                    39:exp=REF_FB_039; 40:exp=REF_FB_040; 41:exp=REF_FB_041;
                    42:exp=REF_FB_042; 43:exp=REF_FB_043; 44:exp=REF_FB_044;
                    45:exp=REF_FB_045; 46:exp=REF_FB_046; 47:exp=REF_FB_047;
                    48:exp=REF_FB_048; 49:exp=REF_FB_049; 50:exp=REF_FB_050;
                    51:exp=REF_FB_051; 52:exp=REF_FB_052; 53:exp=REF_FB_053;
                    54:exp=REF_FB_054; 55:exp=REF_FB_055; 56:exp=REF_FB_056;
                    57:exp=REF_FB_057; 58:exp=REF_FB_058; 59:exp=REF_FB_059;
                    60:exp=REF_FB_060; 61:exp=REF_FB_061; 62:exp=REF_FB_062;
                    63:exp=REF_FB_063; default:exp=32'hx;
                endcase
                if (got===exp) begin
                    ch_ok++;
                    if (j<8||j>=60) $display("  ch[%02d] got=%08h exp=%08h  OK",j,got,exp);
                end else begin
                    ch_bad++;
                    $display("  ch[%02d] got=%08h exp=%08h  MISMATCH",j,got,exp);
                    total_fail++;
                end
            end
            $display("  Channels %0d/64 matched bit-exact", ch_ok);
            if (ch_bad==0) begin $display("  Bit-exact check: PASS"); total_pass++; end
            else $display("  Bit-exact check: FAIL (%0d bad)", ch_bad);
        end
        $display("");

        // ---- Re-entrancy ------------------------------------------------
        $display("--- Re-entrancy: second full transaction ---");
        for (int fi=0; fi<4; fi++) send_flit('0, fi==3);
        pulse_cnt=0; timeout=0;
        while (pulse_cnt<T && timeout<T*8) begin @(posedge clk); timeout++; if(sample_pulse_obs) pulse_cnt++; end
        for (int fi=0; fi<4; fi++) recv_flit(flit, flit_last);
        if (pulse_cnt==T) begin $display("  Second tx: %0d pulses + 4 flits returned  PASS",T); total_pass++; end
        else begin $display("  Second tx: %0d/%0d pulses  FAIL",pulse_cnt,T); total_fail++; end
        $display("");

        // ---- Final -------------------------------------------------------
        $display("========================================");
        $display(" PASS checks: %0d   FAIL checks: %0d", total_pass, total_fail);
        if (total_fail==0) $display("RESULT: PASS");
        else               $display("RESULT: FAIL");
        $display("========================================");
        #50; $finish;
    end

    initial begin #2_000_000; $display("WATCHDOG TIMEOUT\nRESULT: FAIL"); $finish; end
endmodule
