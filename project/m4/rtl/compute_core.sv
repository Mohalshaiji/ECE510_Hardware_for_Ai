// =============================================================================
// compute_core.sv  --  RC-DFA Reservoir Accelerator Digital FSM
// ECE 510 Spring 2026  --  Milestone 4
// Yosys-compatible: no automatic functions, no unsupported SV constructs.
//
// Changes from M3:
//   Task 3 — enable-pipeline for clock tree power reduction.
//             The internal start_r register captures `start` one cycle early,
//             so the entire downstream fanout cone (step_cnt load, cstate
//             transition, sample_pulse assertion) receives the enable with a
//             full 10 ns clock period of margin.  This allows CTS/Yosys to
//             use lower-drive-strength buffers (buf_4/buf_2 instead of buf_12)
//             on those nets, targeting ~60% clock-tree power reduction.
//             Functional behavior is identical: C_IDLE waits for start_r
//             instead of start; latency increases by exactly 1 cycle per call.
// =============================================================================
`default_nettype none
`timescale 1ns/1ps

module compute_core #(
    parameter integer N_IN  = 4096,
    parameter integer N_RES = 128,
    parameter integer N_OUT = 64,
    parameter integer T     = 20
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [N_IN*32-1:0]       e_in,
    input  wire [N_RES*32-1:0]      adc_out,
    output reg                      sample_pulse,
    output wire [N_OUT*32-1:0]      fb_out,
    output reg                      done
);

    // State encoding (2 bits)
    localparam [1:0] C_IDLE   = 2'd0,
                     C_SAMPLE = 2'd1,
                     C_WAIT   = 2'd2,
                     C_DONE   = 2'd3;

    reg [1:0]  cstate;
    reg [4:0]  step_cnt;   // 0..T-1, needs 5 bits for T up to 31
    reg [N_OUT*32-1:0] fb_r;

    // Task 3: registered start — gives CTS a full cycle of margin on the
    // enable fanout cone, reducing required buffer drive strength.
    reg        start_r;

    integer    j;

    always @(posedge clk) begin
        if (!rst_n) begin
            cstate       <= C_IDLE;
            step_cnt     <= 5'd0;
            sample_pulse <= 1'b0;
            done         <= 1'b0;
            fb_r         <= {(N_OUT*32){1'b0}};
            start_r      <= 1'b0;
        end else begin
            sample_pulse <= 1'b0;
            done         <= 1'b0;

            // Register start one cycle ahead of consumption
            start_r <= start;

            case (cstate)
                C_IDLE: begin
                    if (start_r) begin
                        step_cnt <= 5'd0;
                        cstate   <= C_SAMPLE;
                    end
                end

                C_SAMPLE: begin
                    sample_pulse <= 1'b1;
                    cstate       <= C_WAIT;
                end

                C_WAIT: begin
                    step_cnt <= step_cnt + 5'd1;
                    if (step_cnt == T - 1) begin
                        // NODE_IDX gather: fb_r[j] = adc_out[j % N_RES]
                        // For N_OUT=64, N_RES=128: node_idx(j) = j (identity)
                        for (j = 0; j < N_OUT; j = j + 1)
                            fb_r[j*32 +: 32] <= adc_out[(j % N_RES)*32 +: 32];
                        cstate <= C_DONE;
                    end else begin
                        cstate <= C_SAMPLE;
                    end
                end

                C_DONE: begin
                    done   <= 1'b1;
                    cstate <= C_IDLE;
                end

                default: cstate <= C_IDLE;
            endcase
        end
    end

    assign fb_out = fb_r;

endmodule
