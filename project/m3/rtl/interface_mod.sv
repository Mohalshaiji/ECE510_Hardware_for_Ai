// =============================================================================
// interface_mod.sv  --  UCIe AXI-Stream Wrapper
// ECE 510 Spring 2026  --  Milestone 3
// Verilog-2005 compatible for Yosys 0.33
// =============================================================================
`default_nettype none
`timescale 1ns/1ps

module interface_mod #(
    parameter integer N_IN  = 64,
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

    localparam [2:0] S_IDLE    = 3'd0,
                     S_RECV    = 3'd1,
                     S_COMPUTE = 3'd2,
                     S_SEND    = 3'd3;

    reg [2:0] state;
    reg [1:0] flit_cnt;
    reg [N_OUT*32-1:0] fb_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            flit_cnt      <= 2'd0;
            core_start    <= 1'b0;
            s_axis_tready <= 1'b1;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            m_axis_tdata  <= {512{1'b0}};
            m_axis_tkeep  <= {64{1'b1}};
            core_e_in     <= {(N_IN*32){1'b0}};
            fb_r          <= {(N_OUT*32){1'b0}};
        end else begin
            core_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    s_axis_tready <= 1'b1;
                    flit_cnt      <= 2'd0;
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                    if (s_axis_tvalid) begin
                        core_e_in[511:0] <= s_axis_tdata;
                        flit_cnt         <= 2'd1;
                        if (s_axis_tlast) begin
                            core_start    <= 1'b1;
                            s_axis_tready <= 1'b0;
                            state         <= S_COMPUTE;
                        end else begin
                            state <= S_RECV;
                        end
                    end
                end

                S_RECV: begin
                    s_axis_tready <= 1'b1;
                    if (s_axis_tvalid) begin
                        case (flit_cnt)
                            2'd1: core_e_in[1023:512]  <= s_axis_tdata;
                            2'd2: core_e_in[1535:1024] <= s_axis_tdata;
                            2'd3: core_e_in[2047:1536] <= s_axis_tdata;
                            default: ;
                        endcase
                        flit_cnt <= flit_cnt + 2'd1;
                        if (s_axis_tlast) begin
                            core_start    <= 1'b1;
                            s_axis_tready <= 1'b0;
                            state         <= S_COMPUTE;
                        end
                    end
                end

                S_COMPUTE: begin
                    s_axis_tready <= 1'b0;
                    if (core_done) begin
                        fb_r          <= core_fb_out;
                        flit_cnt      <= 2'd0;
                        m_axis_tdata  <= core_fb_out[511:0];
                        m_axis_tkeep  <= {64{1'b1}};
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast  <= 1'b0;
                        state         <= S_SEND;
                    end
                end

                S_SEND: begin
                    if (m_axis_tready && m_axis_tvalid) begin
                        flit_cnt <= flit_cnt + 2'd1;
                        case (flit_cnt + 2'd1)
                            2'd1: begin
                                m_axis_tdata <= fb_r[1023:512];
                                m_axis_tlast <= 1'b0;
                            end
                            2'd2: begin
                                m_axis_tdata <= fb_r[1535:1024];
                                m_axis_tlast <= 1'b0;
                            end
                            2'd3: begin
                                m_axis_tdata <= fb_r[2047:1536];
                                m_axis_tlast <= 1'b1;
                            end
                            default: begin
                                m_axis_tvalid <= 1'b0;
                                m_axis_tlast  <= 1'b0;
                                state         <= S_IDLE;
                            end
                        endcase
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
