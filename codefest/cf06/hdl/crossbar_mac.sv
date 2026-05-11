// =============================================================================
//  bw_crossbar_mac.sv
//  4×4 Binary-Weight Crossbar MAC Unit
//
//  Description:
//    Each weight w[i][j] ∈ {+1, −1}, stored as a single bit:
//      bit = 1  →  weight = +1
//      bit = 0  →  weight = −1
//
//    Every rising clock edge (when en=1):
//      out[j] += Σ_i ( weight[i][j] × in[i] )   for j = 0..3
//
//    Ports
//    ------
//    clk        : clock
//    rst_n      : active-low synchronous reset
//    en         : compute enable (accumulate when high)
//    weight_we  : weight write-enable
//    weight_row : row   index for weight write  (2-bit)
//    weight_col : column index for weight write (2-bit)
//    weight_val : new weight bit (1 = +1, 0 = -1)
//    data_in    : 4 × 8-bit signed inputs
//    accum_out  : 4 × 20-bit signed accumulators
//    accum_clr  : synchronous clear of accumulators
// =============================================================================

module crossbar_mac #(
    parameter int N        = 4,   // crossbar dimension
    parameter int IN_W     = 8,   // input word width (signed)
    parameter int ACCUM_W  = 20   // accumulator width
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,

    // --- weight programming interface ---
    input  logic                         weight_we,
    input  logic [$clog2(N)-1:0]         weight_row,
    input  logic [$clog2(N)-1:0]         weight_col,
    input  logic                         weight_val,   // 1=+1, 0=-1

    // --- data path ---
    input  logic signed [IN_W-1:0]       data_in  [N],
    output logic signed [ACCUM_W-1:0]    accum_out[N],
    input  logic                         accum_clr     // sync clear accumulators
);

    // -------------------------------------------------------------------------
    // Weight register array  [row][col]
    // -------------------------------------------------------------------------
    logic weight_mem [N][N];   // 1 = +1,  0 = -1

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    weight_mem[r][c] <= 1'b1;   // reset to +1
        end else if (weight_we) begin
            weight_mem[weight_row][weight_col] <= weight_val;
        end
    end

    // -------------------------------------------------------------------------
    // Combinational MAC array
    //   partial[j] = Σ_i ( (weight[i][j] ? +in[i] : -in[i]) )
    // -------------------------------------------------------------------------
    // Partial sum width: IN_W bits per term, N terms → needs IN_W + $clog2(N)
    localparam int PSUM_W = IN_W + $clog2(N) + 1;   // +1 for sign safety

    logic signed [PSUM_W-1:0] partial [N];

    always_comb begin
        for (int j = 0; j < N; j++) begin
            partial[j] = '0;
            for (int i = 0; i < N; i++) begin
                partial[j] = partial[j] +
                    ( weight_mem[i][j]
                        ?  PSUM_W'(data_in[i])          // +1 × in[i]
                        : -PSUM_W'(data_in[i]) );        // -1 × in[i]
            end
        end
    end

    // -------------------------------------------------------------------------
    // Accumulator registers
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int j = 0; j < N; j++)
                accum_out[j] <= '0;
        end else if (accum_clr) begin
            for (int j = 0; j < N; j++)
                accum_out[j] <= '0;
        end else if (en) begin
            for (int j = 0; j < N; j++)
                accum_out[j] <= accum_out[j] + ACCUM_W'(partial[j]);
        end
    end

endmodule
