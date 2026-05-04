// =============================================================================
// tb_interface.sv
// Testbench for interface_mod — UCIe AXI-4 Stream Interface Module
// ECE 510 Spring 2026 — Project Milestone 2
//
// DESCRIPTION
//   Exercises the complete AXI-4 Stream write and read transaction sequence.
//   Verifies all required M2 checklist items:
//
//   T1:  s_axis_tready high after reset (module idle)
//   T2:  m_axis_tvalid low after reset
//   T3:  core_start pulsed after final write flit (TLAST)
//   T4:  core_e_in[0] matches first written word
//   T5:  core_e_in[N_IN-1] matches last written word
//   T6:  s_axis_tready de-asserted while core is computing
//   T7:  m_axis_tvalid asserts after core_done
//   T8:  m_axis_tdata carries correct payload (all 4 flits)
//   T9:  m_axis_tlast asserted on final flit only
//   T10: s_axis_tready re-asserts after TX_DONE (module ready for next write)
//   T11: core_start asserted on second write (re-entrancy)
//
// PASS/FAIL  Testbench prints: RESULT: PASS  or  RESULT: FAIL
//
// SIMULATOR  Icarus Verilog 12.0
//   iverilog -g2012 -o sim/tb_interface.vvp tb/tb_interface.sv rtl/interface.sv
//   vvp sim/tb_interface.vvp
// =============================================================================

`timescale 1ns/1ps

module tb_interface;

  localparam N_IN   = 64;
  localparam N_OUT  = 64;
  localparam DATA_W = 32;
  localparam FLIT_W = 512;
  localparam FLIT_B = FLIT_W/8;
  localparam IN_FLITS  = (N_IN *(DATA_W/8)+FLIT_B-1)/FLIT_B;   // 4
  localparam OUT_FLITS = (N_OUT*(DATA_W/8)+FLIT_B-1)/FLIT_B;   // 4

  reg                      clk, rst_n;
  reg                      s_axis_tvalid;
  wire                     s_axis_tready;
  reg  [FLIT_W-1:0]        s_axis_tdata;
  reg  [FLIT_B-1:0]        s_axis_tkeep;
  reg                      s_axis_tlast;
  wire                     m_axis_tvalid;
  reg                      m_axis_tready;
  wire [FLIT_W-1:0]        m_axis_tdata;
  wire [FLIT_B-1:0]        m_axis_tkeep;
  wire                     m_axis_tlast;
  wire                     core_start;
  wire [N_IN*DATA_W-1:0]   core_e_in;
  reg                      core_done;
  reg  [N_OUT*DATA_W-1:0]  core_fb_out;

  interface_mod #(.N_IN(N_IN),.N_OUT(N_OUT),.DATA_W(DATA_W),.FLIT_W(FLIT_W))
    dut (.s_axis_aclk(clk),.s_axis_aresetn(rst_n),
         .s_axis_tvalid(s_axis_tvalid),.s_axis_tready(s_axis_tready),
         .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),
         .s_axis_tlast(s_axis_tlast),
         .m_axis_tvalid(m_axis_tvalid),.m_axis_tready(m_axis_tready),
         .m_axis_tdata(m_axis_tdata),.m_axis_tkeep(m_axis_tkeep),
         .m_axis_tlast(m_axis_tlast),
         .core_start(core_start),.core_e_in(core_e_in),
         .core_done(core_done),.core_fb_out(core_fb_out));

  initial clk = 0;
  always  #0.5 clk = ~clk;

  // Stimulus
  reg [N_IN*DATA_W-1:0]   e_stimulus;
  reg [N_OUT*DATA_W-1:0]  fb_stimulus;
  reg [N_OUT*DATA_W-1:0]  fb_received;
  integer i;
  initial begin
    for (i = 0; i < N_IN;  i = i+1)
      e_stimulus[i*DATA_W +: DATA_W]  = 32'h3C800000 + i;
    for (i = 0; i < N_OUT; i = i+1)
      fb_stimulus[i*DATA_W +: DATA_W] = 32'hBC800000 + i;
    fb_received  = {(N_OUT*DATA_W){1'b0}};
    core_fb_out  = fb_stimulus;
    core_done    = 0;
    m_axis_tready = 0;
  end

  integer pass_cnt, fail_cnt;
  task check_cond;
    input [127:0] name;
    input cond;
    begin
      if (cond) begin $display("  [PASS] %s", name); pass_cnt = pass_cnt+1; end
      else      begin $display("  [FAIL] %s", name); fail_cnt = fail_cnt+1; end
    end
  endtask

  // Write task: negedge-sampled handshake
  task axi_write;
    input [N_IN*DATA_W-1:0] data;
    integer f;
    reg wr;
    begin
      f = 0;
      while (f < IN_FLITS) begin
        s_axis_tvalid = 1;
        s_axis_tdata  = data[f*FLIT_W +: FLIT_W];
        s_axis_tkeep  = {FLIT_B{1'b1}};
        s_axis_tlast  = (f == IN_FLITS-1);
        @(negedge clk); wr = s_axis_tready;
        @(posedge clk); #0.1;
        if (wr) f = f + 1;
      end
      s_axis_tvalid = 0;
      s_axis_tlast  = 0;
    end
  endtask

  // Read task: posedge-sampled, 1 cycle per flit
  // Starts with tready=1, waits for tvalid
  integer tlast_idx;
  task axi_read_simple;
    output [N_OUT*DATA_W-1:0] result;
    integer f;
    begin
      result   = {(N_OUT*DATA_W){1'b0}};
      tlast_idx = -1;
      m_axis_tready = 1;
      for (f = 0; f < OUT_FLITS; f = f+1) begin
        // Wait for valid at posedge
        while (!m_axis_tvalid) @(posedge clk);
        result[f*FLIT_W +: FLIT_W] = m_axis_tdata;
        if (m_axis_tlast) tlast_idx = f;
        @(posedge clk); #0.1;
      end
      m_axis_tready = 0;
    end
  endtask

  // core_start edge counter
  integer start_edges;
  always @(posedge clk)
    if (core_start) start_edges = start_edges + 1;

  initial begin
    $dumpfile("sim/interface_wave.vcd");
    $dumpvars(0, tb_interface);

    pass_cnt    = 0;
    fail_cnt    = 0;
    start_edges = 0;
    rst_n       = 0;
    s_axis_tvalid = 0;
    s_axis_tdata  = {FLIT_W{1'b0}};
    s_axis_tkeep  = {FLIT_B{1'b1}};
    s_axis_tlast  = 0;

    $display("=== tb_interface: UCIe AXI-S Interface Module ===");
    $display("    N_IN=%0d N_OUT=%0d FLIT_W=%0d IN_FLITS=%0d OUT_FLITS=%0d",
             N_IN, N_OUT, FLIT_W, IN_FLITS, OUT_FLITS);

    repeat(3) @(posedge clk); #0.1; rst_n = 1;
    @(posedge clk); #0.1;

    // T1, T2: post-reset state
    check_cond("T1 s_tready=1 after reset",  s_axis_tready == 1);
    check_cond("T2 m_tvalid=0 after reset",  m_axis_tvalid == 0);

    // T3-T6: write transaction
    $display("  --- Write transaction ---");
    start_edges = 0;
    axi_write(e_stimulus);
    @(posedge clk); #0.1;

    check_cond("T3 core_start pulsed after TLAST",   start_edges >= 1);
    check_cond("T4 core_e_in[0] == e_stim[0]",
               core_e_in[DATA_W-1:0] == e_stimulus[DATA_W-1:0]);
    check_cond("T5 core_e_in[N-1] == e_stim[N-1]",
               core_e_in[(N_IN-1)*DATA_W +: DATA_W] ==
               e_stimulus[(N_IN-1)*DATA_W +: DATA_W]);
    check_cond("T6 s_tready=0 while computing",      s_axis_tready == 0);

    // T7-T9: read transaction
    $display("  --- Read transaction ---");
    repeat(5) @(posedge clk);
    core_done = 1; @(posedge clk); #0.1; core_done = 0;

    // Wait up to 10 cycles for m_axis_tvalid
    begin : wait_mv
      integer wc;
      for (wc = 0; wc < 10; wc = wc+1) begin
        if (m_axis_tvalid) disable wait_mv;
        @(posedge clk);
      end
    end

    check_cond("T7 m_tvalid asserts after core_done", m_axis_tvalid == 1);

    axi_read_simple(fb_received);

    check_cond("T8 received payload == fb_stimulus",
               fb_received == fb_stimulus);
    check_cond("T9 TLAST on final flit (index 3)",
               tlast_idx == OUT_FLITS-1);

    // T10, T11: second cycle
    $display("  --- Second write-read cycle ---");
    @(posedge clk); #0.1;
    check_cond("T10 s_tready re-asserts after TX_DONE", s_axis_tready == 1);

    start_edges = 0;
    axi_write(e_stimulus);
    @(posedge clk); #0.1;
    check_cond("T11 core_start on second write", start_edges >= 1);

    // Summary
    $display("--- %0d / %0d checks passed ---",
             pass_cnt, pass_cnt + fail_cnt);
    if (fail_cnt == 0)
      $display("RESULT: PASS");
    else
      $display("RESULT: FAIL");

    #20; $finish;
  end

  initial begin
    #2000;
    $display("RESULT: FAIL (global timeout)");
    $finish;
  end

endmodule
