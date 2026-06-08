# pnr_constraints.sdc -- clock and IO constraints for OpenROAD PnR
# RC-DFA M4 | ECE 510 Spring 2026
# Analog arrival time reused from M3 ngspice characterisation (t_99 = 33.380 ps).
create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay  0.5 -clock clk [get_ports {rst_n s_axis_tdata s_axis_tvalid s_axis_tlast s_axis_tkeep m_axis_tready}]
set_output_delay 0.5 -clock clk [get_ports {s_axis_tready m_axis_tdata m_axis_tvalid m_axis_tlast m_axis_tkeep sample_pulse}]
set_input_delay 0.033380 -clock clk [get_ports {adc_out}]
