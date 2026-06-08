// =============================================================================
// top.sv  --  RC-DFA Memcapacitive Reservoir Accelerator Integration Top
// ECE 510 Spring 2026  --  Milestone 4
// Verilog-2005 compatible for Yosys 0.33
//
// Changes from M3:
//   - N_IN default changed from 64 to 4096 (Task 1: interface widening).
//     The input bus now carries one full spatial tile row: 4096 FP32 channels
//     = 131,072 bits = 256 x 512-bit AXI-S flits.
//   - interface_mod renamed to interface_mod (file: interface.sv) with
//     updated N_IN parameter.  All other ports unchanged.
//
// Ports:
//   clk            input   1b    100 MHz system clock
//   rst_n          input   1b    active-low synchronous reset
//   s_axis_tdata   input  512b   UCIe flit from host
//   s_axis_tvalid  input   1b    AXI-S valid
//   s_axis_tready  output  1b    AXI-S ready
//   s_axis_tlast   input   1b    last flit of input vector
//   s_axis_tkeep   input  64b    byte enable
//   m_axis_tdata   output 512b   UCIe flit to host
//   m_axis_tvalid  output  1b    AXI-S valid
//   m_axis_tready  input   1b    AXI-S ready
//   m_axis_tlast   output  1b    last flit of output vector
//   m_axis_tkeep   output 64b    byte enable
//   adc_out        input 4096b   FP32 ADC readback from analog crossbar
//                                (128 nodes x 32b)
//   sample_pulse   output  1b    trigger to analog crossbar
// =============================================================================
`default_nettype none
`timescale 1ns/1ps

module top #(
    parameter integer N_IN  = 4096,
    parameter integer N_RES = 128,
    parameter integer N_OUT = 64,
    parameter integer T     = 20
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [511:0]             s_axis_tdata,
    input  wire                     s_axis_tvalid,
    output wire                     s_axis_tready,
    input  wire                     s_axis_tlast,
    input  wire [63:0]              s_axis_tkeep,

    output wire [511:0]             m_axis_tdata,
    output wire                     m_axis_tvalid,
    input  wire                     m_axis_tready,
    output wire                     m_axis_tlast,
    output wire [63:0]              m_axis_tkeep,

    input  wire [N_RES*32-1:0]      adc_out,
    output wire                     sample_pulse
);

    wire [N_IN*32-1:0]   e_in_w;
    wire [N_OUT*32-1:0]  fb_out_w;
    wire                 core_start;
    wire                 core_done;

    interface_mod #(
        .N_IN (N_IN),
        .N_OUT(N_OUT)
    ) u_iface (
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
        .core_e_in     (e_in_w),
        .core_start    (core_start),
        .core_fb_out   (fb_out_w),
        .core_done     (core_done)
    );

    compute_core #(
        .N_IN (N_IN),
        .N_RES(N_RES),
        .N_OUT(N_OUT),
        .T    (T)
    ) u_core (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (core_start),
        .e_in         (e_in_w),
        .adc_out      (adc_out),
        .sample_pulse (sample_pulse),
        .fb_out       (fb_out_w),
        .done         (core_done)
    );

endmodule
