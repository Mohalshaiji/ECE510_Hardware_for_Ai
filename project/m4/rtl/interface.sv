// =============================================================================
// interface.sv  --  UCIe AXI-Stream Wrapper
// ECE 510 Spring 2026  --  Milestone 4
// Verilog-2005 compatible for Yosys 0.33
//
// Changes from M3 interface_mod.sv:
//   Task 1 — Generic flit count.  N_IN is now a parameter; the number of
//             512-bit flits is N_FLITS = (N_IN*32)/512 = N_IN/16.  For the
//             M4 default of N_IN=4096 this is 256 flits in and 256 flits out.
//
//   Task 2 — Double-buffer pipeline.  Two e_in register banks (buf_A, buf_B)
//             and a fill pointer (buf_fill) track which bank is being written.
//             s_axis_tready is held HIGH during S_COMPUTE so the host can
//             begin streaming the next input immediately.
//
//   Task 2 rev2 — Replaced for-loop comparator tree with direct dynamic
//             bit-select: buf[flit_cnt*512 +: 512] <= s_axis_tdata.
//             Eliminates 4,102 o211a_2 comparator cells from synthesis,
//             replacing them with a simple write-enable mux. Removed unused
//             buf_active register and integer wi loop variable.
//
// Flit arithmetic (N_IN=4096, FP32):
//   Input  payload : 4096 x 32b = 131,072 b  -> 256 x 512b flits
//   Output payload : N_OUT x 32b (N_OUT=64)  ->   4 x 512b flits
// =============================================================================
`default_nettype none
`timescale 1ns/1ps

module interface_mod #(
    parameter integer N_IN  = 4096,
    parameter integer N_OUT = 64
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [511:0]             s_axis_tdata,
    input  wire                     s_axis_tvalid,
    output reg                      s_axis_tready,
    input  wire                     s_axis_tlast,
    input  wire [63:0]              s_axis_tkeep,

    output reg  [511:0]             m_axis_tdata,
    output reg                      m_axis_tvalid,
    input  wire                     m_axis_tready,
    output reg                      m_axis_tlast,
    output reg  [63:0]              m_axis_tkeep,

    output reg  [N_IN*32-1:0]       core_e_in,
    output reg                      core_start,
    input  wire [N_OUT*32-1:0]      core_fb_out,
    input  wire                     core_done
);

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------
    localparam integer FLIT_BITS   = 512;
    localparam integer N_FLITS_IN  = (N_IN  * 32) / FLIT_BITS;  // 256 for N_IN=4096
    localparam integer N_FLITS_OUT = (N_OUT * 32) / FLIT_BITS;  //   4 for N_OUT=64
    localparam integer CNT_W       = 9;  // ceil(log2(256)) = 8, +1 guard

    // -----------------------------------------------------------------------
    // State encoding
    // -----------------------------------------------------------------------
    localparam [1:0] S_IDLE    = 2'd0,
                     S_RECV    = 2'd1,
                     S_COMPUTE = 2'd2,
                     S_SEND    = 2'd3;

    reg [1:0]          state;
    reg [CNT_W-1:0]    flit_cnt;
    reg [N_OUT*32-1:0] fb_r;

    // -----------------------------------------------------------------------
    // Task 2 — double-buffer: two input banks, one fill pointer
    // buf_active removed (was unused)
    // -----------------------------------------------------------------------
    reg [N_IN*32-1:0] buf_A;    // input bank A
    reg [N_IN*32-1:0] buf_B;    // input bank B
    reg               buf_fill; // 0 = filling A, 1 = filling B

    // -----------------------------------------------------------------------
    // Write helper: write one 512-bit flit into the correct bank slot.
    // Uses dynamic bit-select — no comparator tree, no loop variable FF.
    // -----------------------------------------------------------------------
    // write address in bits = flit_cnt * 512
    wire [17:0] waddr = {flit_cnt, 9'd0};  // flit_cnt << 9

    // -----------------------------------------------------------------------
    // Send mux: 4-way (N_FLITS_OUT=4), small enough for static case
    // -----------------------------------------------------------------------
    integer si;

    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            flit_cnt      <= {CNT_W{1'b0}};
            core_start    <= 1'b0;
            s_axis_tready <= 1'b1;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            m_axis_tdata  <= {512{1'b0}};
            m_axis_tkeep  <= {64{1'b1}};
            core_e_in     <= {(N_IN*32){1'b0}};
            fb_r          <= {(N_OUT*32){1'b0}};
            buf_A         <= {(N_IN*32){1'b0}};
            buf_B         <= {(N_IN*32){1'b0}};
            buf_fill      <= 1'b0;
        end else begin
            core_start <= 1'b0;

            case (state)

                // -----------------------------------------------------------
                // S_IDLE: wait for first valid flit
                // -----------------------------------------------------------
                S_IDLE: begin
                    s_axis_tready <= 1'b1;
                    flit_cnt      <= {CNT_W{1'b0}};
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                    buf_fill      <= 1'b0;
                    if (s_axis_tvalid) begin
                        // Flit 0 into buf_A slot 0
                        buf_A[511:0] <= s_axis_tdata;
                        flit_cnt     <= {{(CNT_W-1){1'b0}}, 1'b1};
                        if (s_axis_tlast) begin
                            core_e_in  <= buf_A;
                            core_start <= 1'b1;
                            state      <= S_COMPUTE;
                        end else begin
                            state <= S_RECV;
                        end
                    end
                end

                // -----------------------------------------------------------
                // S_RECV: accumulate remaining flits into fill bank.
                // Direct bit-select replaces 256-way comparator loop.
                // -----------------------------------------------------------
                S_RECV: begin
                    s_axis_tready <= 1'b1;
                    if (s_axis_tvalid) begin
                        // Write flit at dynamic offset flit_cnt*512
                        if (!buf_fill)
                            buf_A[waddr +: 512] <= s_axis_tdata;
                        else
                            buf_B[waddr +: 512] <= s_axis_tdata;

                        flit_cnt <= flit_cnt + {{(CNT_W-1){1'b0}}, 1'b1};

                        if (s_axis_tlast) begin
                            core_e_in  <= (!buf_fill) ? buf_A : buf_B;
                            core_start <= 1'b1;
                            flit_cnt   <= {CNT_W{1'b0}};
                            buf_fill   <= ~buf_fill;
                            state      <= S_COMPUTE;
                        end
                    end
                end

                // -----------------------------------------------------------
                // S_COMPUTE: core running; accept next call into other bank.
                // -----------------------------------------------------------
                S_COMPUTE: begin
                    s_axis_tready <= 1'b1;

                    if (s_axis_tvalid) begin
                        if (!buf_fill)
                            buf_A[waddr +: 512] <= s_axis_tdata;
                        else
                            buf_B[waddr +: 512] <= s_axis_tdata;
                        flit_cnt <= flit_cnt + {{(CNT_W-1){1'b0}}, 1'b1};
                    end

                    if (core_done) begin
                        fb_r          <= core_fb_out;
                        flit_cnt      <= {CNT_W{1'b0}};
                        m_axis_tdata  <= core_fb_out[511:0];
                        m_axis_tkeep  <= {64{1'b1}};
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast  <= (N_FLITS_OUT == 1) ? 1'b1 : 1'b0;
                        state         <= S_SEND;
                    end
                end

                // -----------------------------------------------------------
                // S_SEND: stream output flits back to host (N_FLITS_OUT=4)
                // -----------------------------------------------------------
                S_SEND: begin
                    if (m_axis_tready && m_axis_tvalid) begin
                        flit_cnt <= flit_cnt + {{(CNT_W-1){1'b0}}, 1'b1};
                        if (flit_cnt + 1 >= N_FLITS_OUT[CNT_W-1:0]) begin
                            m_axis_tvalid <= 1'b0;
                            m_axis_tlast  <= 1'b0;
                            flit_cnt      <= {CNT_W{1'b0}};
                            state         <= S_IDLE;
                        end else begin
                            for (si = 1; si < N_FLITS_OUT; si = si + 1) begin
                                if (flit_cnt + 1 == si[CNT_W-1:0]) begin
                                    m_axis_tdata <= fb_r[si*512 +: 512];
                                    m_axis_tlast <= (si == N_FLITS_OUT - 1) ? 1'b1 : 1'b0;
                                end
                            end
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
