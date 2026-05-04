// =============================================================================
// interface.sv
// UCIe AXI-4 Stream Interface Module — RC-DFA Reservoir Accelerator
// ECE 510 Spring 2026 — Project Milestone 2
//
// PURPOSE
//   Die-to-die framing between host CPU (UCIe physical layer) and the
//   memcapacitive reservoir compute core. Selected interface: UCIe,
//   justified in project/m1/interface_selection.md.
//
//   Implements AXI4-Stream on the UCIe logical data layer
//   (UCIe 1.1 Specification, Section 6.3, Streaming Protocol).
//   512-bit (64-byte) flit width matches UCIe D2D adapter width.
//
// TRANSACTION FORMAT
//   Write (host -> chiplet, error map in):
//     Host drives TVALID; module asserts TREADY when idle.
//     Each flit: FLIT_W=512 bits. N_IN=64 FP32 words = 256 bytes = 4 flits.
//     TLAST on final flit. After TLAST: core_start pulsed, core_e_in valid.
//
//   Read (chiplet -> host, feedback map out):
//     After core_done, module drives m_axis_tvalid.
//     N_OUT FP32 words packed into 4 flits; TLAST on final flit.
//     Respects m_axis_tready backpressure.
//
// UCIe SPEC REFERENCE
//   UCIe 1.1, Section 6.3: D2D Adapter Streaming Channel.
//   TDATA width = 512 bits per flit. Latency: sub-nanosecond (physical).
//
// REGISTER MAP
//   Streaming protocol; no addressable CSR space.
//   Channel 0 (s_axis_*): error map ingress.
//   Channel 1 (m_axis_*): feedback map egress.
//
// PARAMETERS
//   N_IN    : input error channels (64)
//   N_OUT   : output feedback channels (64)
//   DATA_W  : bits per word (32)
//   FLIT_W  : UCIe flit width (512)
//
// PORTS
//   s_axis_aclk     : in  1        AXI-S clock (shared)
//   s_axis_aresetn  : in  1        Active-low synchronous reset
//   s_axis_tvalid   : in  1        Error map flit valid (host)
//   s_axis_tready   : out 1        Module ready to accept
//   s_axis_tdata    : in  FLIT_W   Flit payload (FP32 packed)
//   s_axis_tkeep    : in  FLIT_W/8 Byte enable
//   s_axis_tlast    : in  1        Last flit of vector
//   m_axis_tvalid   : out 1        Feedback flit valid
//   m_axis_tready   : in  1        Host ready to accept
//   m_axis_tdata    : out FLIT_W   Flit payload
//   m_axis_tkeep    : out FLIT_W/8 All ones (all bytes valid)
//   m_axis_tlast    : out 1        Last flit of feedback vector
//   core_start      : out 1        Pulse to start compute core
//   core_e_in       : out N_IN*32  Assembled error vector to core
//   core_done       : in  1        Core computation complete
//   core_fb_out     : in  N_OUT*32 Feedback vector from core
//
// CLOCK/RESET
//   Single clock domain: s_axis_aclk. Synchronous active-low reset.
// =============================================================================

module interface_mod
  #(parameter N_IN   = 64,
    parameter N_OUT  = 64,
    parameter DATA_W = 32,
    parameter FLIT_W = 512)
   (input  wire                    s_axis_aclk,
    input  wire                    s_axis_aresetn,
    input  wire                    s_axis_tvalid,
    output reg                     s_axis_tready,
    input  wire [FLIT_W-1:0]       s_axis_tdata,
    input  wire [FLIT_W/8-1:0]     s_axis_tkeep,
    input  wire                    s_axis_tlast,
    output reg                     m_axis_tvalid,
    input  wire                    m_axis_tready,
    output reg  [FLIT_W-1:0]       m_axis_tdata,
    output reg  [FLIT_W/8-1:0]     m_axis_tkeep,
    output reg                     m_axis_tlast,
    output reg                     core_start,
    output reg  [N_IN*DATA_W-1:0]  core_e_in,
    input  wire                    core_done,
    input  wire [N_OUT*DATA_W-1:0] core_fb_out);

  // --------------------------------------------------------------------------
  // Derived constants
  // --------------------------------------------------------------------------
  localparam VEC_BYTES_IN  = N_IN  * (DATA_W/8);          // 256
  localparam VEC_BYTES_OUT = N_OUT * (DATA_W/8);          // 256
  localparam FLIT_BYTES    = FLIT_W / 8;                  // 64
  localparam IN_FLITS  = (VEC_BYTES_IN  + FLIT_BYTES-1) / FLIT_BYTES;  // 4
  localparam OUT_FLITS = (VEC_BYTES_OUT + FLIT_BYTES-1) / FLIT_BYTES;  // 4
  localparam FCNT_W = 3;                                  // covers 0..3

  // --------------------------------------------------------------------------
  // FSM states
  // --------------------------------------------------------------------------
  localparam [2:0] RX_IDLE = 3'd0,
                   RX_RECV = 3'd1,
                   RX_WAIT = 3'd2,
                   TX_SEND = 3'd3,
                   TX_DONE = 3'd4;

  reg [2:0]              state_q, state_d;
  reg [FCNT_W-1:0]       rx_flit_cnt_q;  // RX phase counter
  reg [FCNT_W-1:0]       tx_flit_cnt_q;  // TX phase counter
  reg [N_IN*DATA_W-1:0]  e_buf_q;
  reg [N_OUT*DATA_W-1:0] fb_buf_q;

  // --------------------------------------------------------------------------
  // Sequential
  // --------------------------------------------------------------------------
  always @(posedge s_axis_aclk) begin
    if (!s_axis_aresetn) begin
      state_q    <= RX_IDLE;
      rx_flit_cnt_q <= {FCNT_W{1'b0}};
      tx_flit_cnt_q <= {FCNT_W{1'b0}};
      e_buf_q    <= {(N_IN*DATA_W){1'b0}};
      fb_buf_q   <= {(N_OUT*DATA_W){1'b0}};
    end else begin
      state_q <= state_d;

      // Receive: pack incoming flits into e_buf
      if (s_axis_tvalid && s_axis_tready) begin
        case (rx_flit_cnt_q)
          3'd0: e_buf_q[  0*FLIT_W +: FLIT_W] <= s_axis_tdata;
          3'd1: e_buf_q[  1*FLIT_W +: FLIT_W] <= s_axis_tdata;
          3'd2: e_buf_q[  2*FLIT_W +: FLIT_W] <= s_axis_tdata;
          3'd3: e_buf_q[  3*FLIT_W +: FLIT_W] <= s_axis_tdata;
          default: ;
        endcase
      end

      // Capture feedback when core done
      if (state_q == RX_WAIT && core_done)
        fb_buf_q <= core_fb_out;

      // RX flit counter
      if (state_d == RX_IDLE || state_d == RX_WAIT)
        rx_flit_cnt_q <= {FCNT_W{1'b0}};
      else if ((state_q == RX_RECV || state_q == RX_IDLE) &&
               s_axis_tvalid && s_axis_tready) begin
        if (s_axis_tlast) rx_flit_cnt_q <= {FCNT_W{1'b0}};
        else              rx_flit_cnt_q <= rx_flit_cnt_q + 1;
      end
      // TX flit counter
      if (state_d == TX_SEND && state_q != TX_SEND)
        tx_flit_cnt_q <= {FCNT_W{1'b0}};
      else if (state_q == TX_SEND && m_axis_tvalid && m_axis_tready) begin
        if (m_axis_tlast) tx_flit_cnt_q <= {FCNT_W{1'b0}};
        else              tx_flit_cnt_q <= tx_flit_cnt_q + 1;
      end
    end
  end

  // --------------------------------------------------------------------------
  // FSM next-state + combinational outputs
  // --------------------------------------------------------------------------
  reg [FLIT_W-1:0]   tx_flit;

  always @(*) begin
    state_d       = state_q;
    s_axis_tready = 1'b0;
    m_axis_tvalid = 1'b0;
    m_axis_tlast  = 1'b0;
    m_axis_tkeep  = {(FLIT_W/8){1'b1}};
    core_start    = 1'b0;
    core_e_in     = e_buf_q;

    // Mux the outgoing flit from fb_buf_q
    case (tx_flit_cnt_q)
      3'd0: tx_flit = fb_buf_q[0*FLIT_W +: FLIT_W];
      3'd1: tx_flit = fb_buf_q[1*FLIT_W +: FLIT_W];
      3'd2: tx_flit = fb_buf_q[2*FLIT_W +: FLIT_W];
      3'd3: tx_flit = fb_buf_q[3*FLIT_W +: FLIT_W];
      default: tx_flit = {FLIT_W{1'b0}};
    endcase
    m_axis_tdata = tx_flit;

    case (state_q)
      RX_IDLE: begin
        s_axis_tready = 1'b1;
        if (s_axis_tvalid) state_d = RX_RECV;
      end
      RX_RECV: begin
        s_axis_tready = 1'b1;
        if (s_axis_tvalid && s_axis_tlast) begin
          core_start = 1'b1;
          state_d    = RX_WAIT;
        end
      end
      RX_WAIT: begin
        if (core_done) state_d = TX_SEND;
      end
      TX_SEND: begin
        m_axis_tvalid = 1'b1;
        m_axis_tlast  = (tx_flit_cnt_q == 3'(OUT_FLITS - 1));
        if (m_axis_tready && m_axis_tlast)
          state_d = TX_DONE;
      end
      TX_DONE: state_d = RX_IDLE;
      default: state_d = RX_IDLE;
    endcase
  end

endmodule
