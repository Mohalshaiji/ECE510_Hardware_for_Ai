//Generated with Qwen2.5Coder:7b
module mac (
    input  logic          clk,
    input  logic          rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 0;
        end else begin
            out <= out + (a * b);
        end
    end
endmodule
