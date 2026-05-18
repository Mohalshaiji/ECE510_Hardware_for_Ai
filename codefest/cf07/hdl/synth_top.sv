// =============================================================================
//  crossbar_mac (sv2v Converted Verilog)
//  4x4 Binary-Weight Crossbar MAC Unit
// =============================================================================

module crossbar_mac (
    clk,
    rst_n,
    en,
    weight_we,
    weight_row,
    weight_col,
    weight_val,
    data_in,
    accum_out,
    accum_clr
);
    reg _sv2v_0; // Internal flag used by sv2v to manage initialization/simulation behavior
    
    // --- Architectural Parameters ---
    parameter N = 4;                        // Crossbar dimension (NxN)
    parameter IN_W = 8;                     // Input word width (signed)
    parameter ACCUM_W = 20;                 // Accumulator width
    parameter CLOG2_N = 2;                  // Precomputed ceiling log2(N)
    parameter PSUM_W = (IN_W + CLOG2_N) + 1;// Partial sum width to avoid sign/overflow issues

    // --- Input / Output Ports ---
    input wire clk;                         // Clock signal
    input wire rst_n;                       // Active-low synchronous reset
    input wire en;                          // Compute enable (accumulates when high)
    input wire weight_we;                   // Weight write-enable
    input wire [CLOG2_N - 1:0] weight_row;  // Row index for weight programming
    input wire [CLOG2_N - 1:0] weight_col;  // Column index for weight programming
    input wire weight_val;                  // New weight bit value (1 = +1, 0 = -1)
    
    // sv2v flattened the SystemVerilog arrays into single-dimension packed vectors
    input wire signed [(N * IN_W) - 1:0] data_in;       // Flattened 4 x 8-bit signed inputs
    output reg signed [(N * ACCUM_W) - 1:0] accum_out; // Flattened 4 x 20-bit signed outputs
    input wire accum_clr;                   // Synchronous clear for accumulators

    // --- Internal Storage ---
    // 2D register array representing the NxN weight matrix
    reg weight_mem [0:N - 1][0:N - 1];

    // --- Weight Register Array Updates ---
    always @(posedge clk)
        if (!rst_n) begin : sv2v_autoblock_1
            reg signed [31:0] r;
            for (r = 0; r < N; r = r + 1)
                begin : sv2v_autoblock_2
                    reg signed [31:0] c;
                    for (c = 0; c < N; c = c + 1)
                        weight_mem[r][c] <= 1'b1; // Reset all weight configurations to +1
                end
        end
        else if (weight_we)
            weight_mem[weight_row][weight_col] <= weight_val; // Write weight bit to specified index

    // --- Combinational MAC Array Logic ---
    // Holds the calculated dot products for the current cycle before accumulation
    reg signed [PSUM_W - 1:0] partial [0:N - 1];

    always @(*) begin
        if (_sv2v_0)
            ;
        begin : sv2v_autoblock_3
            reg signed [31:0] j;
            for (j = 0; j < N; j = j + 1)
                begin
                    partial[j] = {PSUM_W {1'b0}}; // Initialize partial sum to zero
                    begin : sv2v_autoblock_4
                        reg signed [31:0] i;
                        for (i = 0; i < N; i = i + 1)
                            // Computes: partial[j] += (weight_mem[i][j] ? +data_in[i] : -data_in[i])
                            // Uses bit-stream slicing to correctly index into the flattened data_in array.
                            partial[j] = partial[j] + (weight_mem[i][j] ? {{PSUM_W - IN_W {data_in[(((N - 1) - i) * IN_W) + (IN_W - 1)]}}, data_in[((N - 1) - i) * IN_W+:IN_W]} : -{{PSUM_W - IN_W {data_in[(((N - 1) - i) * IN_W) + (IN_W - 1)]}}, data_in[((N - 1) - i) * IN_W+:IN_W]});
                    end
                end
        end
    end

    // --- Accumulator Registers ---
    always @(posedge clk)
        if (!rst_n) begin : sv2v_autoblock_5
            reg signed [31:0] j;
            for (j = 0; j < N; j = j + 1)
                // Reset condition: Clear all packed bits in the accumulator vector
                accum_out[((N - 1) - j) * ACCUM_W+:ACCUM_W] <= {ACCUM_W {1'b0}};
        end
        else if (accum_clr) begin : sv2v_autoblock_6
            reg signed [31:0] j;
            for (j = 0; j < N; j = j + 1)
                // Synchronous clear: Reset accumulator values back to zero
                accum_out[((N - 1) - j) * ACCUM_W+:ACCUM_W] <= {ACCUM_W {1'b0}};
        end
        else if (en) begin : sv2v_autoblock_7
            reg signed [31:0] j;
            for (j = 0; j < N; j = j + 1)
                // Accumulate condition: Sign-extends the partial sum and adds it to the active accumulation index
                accum_out[((N - 1) - j) * ACCUM_W+:ACCUM_W] <= accum_out[((N - 1) - j) * ACCUM_W+:ACCUM_W] + {{ACCUM_W - PSUM_W {partial[j][PSUM_W - 1]}}, partial[j]};
        end

    // --- Initialization block ---
    initial _sv2v_0 = 0;
endmodule
