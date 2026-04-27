// mac_correct.v
// Corrected MAC unit — synthesizable SystemVerilog
// Fixes applied vs LLM outputs:
//   1. Explicit typed reset literal 32'sd0
//   2. Explicit signed intermediate product to make sign-extension unambiguous
//   3. begin/end on all conditional branches

module mac (
    input  logic              clk,
    input  logic              rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);

    // Explicit 16-bit signed product — makes sign-extension to 32 bits auditable
    logic signed [15:0] product;
    assign product = a * b;

    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            out <= out + {{16{product[15]}}, product};  // explicit sign-extension
        end
    end

endmodule
