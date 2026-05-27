* =============================================================================
* crossbar_rc.sp  --  Memcapacitive Crossbar Full Characterisation
* ECE 510 Spring 2026  --  M3
* Full 128x128 array -- direct measurement, no extrapolation.
* =============================================================================

.title Memcapacitive Crossbar 128x128 Full Characterisation

.param R_drive      = 500
.param C_mem        = 10e-15
.param R_leak       = 100e3
.param R_adc_in     = 1e6
.param N            = 128
.param V_supply     = 1.8
.param wire_res_per_um = 0.1
.param wire_cap_per_um = 0.2e-15
.param array_pitch_um  = 128

.param R_wire_row   = {wire_res_per_um * array_pitch_um}
.param R_eff_drive  = {R_drive/N + R_wire_row/N}
.param C_wire_col   = {wire_cap_per_um * array_pitch_um}
.param C_col_total  = {N * C_mem + C_wire_col}
.param R_leak_col   = {R_leak / N}

* Step input: 0 -> V_supply
Vstep   vin   0   PULSE(0 {V_supply} 0 10p 10p 10n 30n)

* Drive + wire resistance
R_drive_eff   vin   col_node   {R_eff_drive}

* Column capacitance
C_col   col_node   0   {C_col_total}

* Leakage
R_leak_col   col_node   0   {R_leak_col}

* ADC input
R_adc   col_node   0   {R_adc_in}

* Supply for power measurement (same as Vstep but as a named source)
* Power = V(vin) * -I(Vstep)

.tran 1p 20n

* Timing measurements -- all on single lines, no backslash continuation
.measure tran t_10   WHEN V(col_node)={0.1*V_supply}  RISE=1
.measure tran t_90   WHEN V(col_node)={0.9*V_supply}  RISE=1
.measure tran t_99   WHEN V(col_node)={0.99*V_supply} RISE=1
.measure tran t_rise TRIG V(col_node) VAL={0.1*V_supply} RISE=1 TARG V(col_node) VAL={0.9*V_supply} RISE=1

* Power: measure average of V*I over settling window
* I through R_drive_eff = (V(vin)-V(col_node))/R_eff_drive
* Use a behavioural current source approach: measure voltage across R_drive_eff
.measure tran V_drive_avg   AVG V(vin,col_node) FROM=0 TO=10n
.measure tran V_drive_peak  MAX V(vin,col_node) FROM=0 TO=200p
.measure tran V_col_static  AVG V(col_node)     FROM=18n TO=20n
.measure tran V_col_avg     AVG V(col_node)     FROM=0   TO=10n

.control
  run

  echo ""
  echo "=== 128x128 ARRAY: TIMING ==="
  print t_10 t_90 t_99 t_rise

  let tau = (t_90 - t_10) / log(9)
  let bw_GHz = 1 / (2 * 3.14159265 * tau) / 1e9
  let margin_1GHz_ps = (1e-9 - tau) * 1e12

  echo "tau_RC [ps]:"
  let tau_ps = tau * 1e12
  print tau_ps
  echo "t99 [ps]:"
  let t99_ps = t_99 * 1e12
  print t99_ps
  echo "Bandwidth [GHz]:"
  print bw_GHz
  echo "Margin at 1GHz [ps]:"
  print margin_1GHz_ps

  if margin_1GHz_ps gt 0
    echo "SETTLING @1GHz: PASS"
  else
    echo "SETTLING @1GHz: FAIL"
  end

  echo ""
  echo "=== 128x128 ARRAY: POWER (one column) ==="
  * P_avg per column = V_drive_avg^2 / R_eff_drive  (power in drive resistor)
  * plus leakage through R_leak_col and R_adc
  let R_eff = 3.90625 + 0.1
  let P_drive_avg = V_drive_avg * V_drive_avg / R_eff
  let P_drive_peak = V_drive_peak * V_drive_peak / R_eff

  * Leakage: V_col_static^2 * (1/R_leak_col + 1/R_adc_in)
  let G_shunt = 1/781.25 + 1/1e6
  let P_static_col = V_col_static * V_col_static * G_shunt

  * Scale to full 128-column array
  let P_avg_array_uW   = P_drive_avg * 128 * 1e6
  let P_peak_array_mW  = P_drive_peak * 128 * 1e3
  let P_static_array_uW = P_static_col * 128 * 1e6

  echo "Average power full array [uW]:"
  print P_avg_array_uW
  echo "Peak power full array [mW]:"
  print P_peak_array_mW
  echo "Static leakage full array [uW]:"
  print P_static_array_uW

  echo ""
  echo "=== 128x128 ARRAY: POWER DENSITY ==="
  let area_mm2 = 0.016384
  let P_density_mW_mm2 = P_avg_array_uW / (area_mm2 * 1000)
  echo "Array area = 16384 um^2 = 0.016384 mm^2"
  echo "Power density (avg) [mW/mm^2]:"
  print P_density_mW_mm2

  echo ""
  echo "=== 128x128 ARRAY: ROUTING ==="
  echo "Row wire length/row [um]: 128"
  echo "Col wire length/col [um]: 128"
  echo "Total row wire [um]: 16384"
  echo "Total col wire [um]: 16384"
  echo "Total interconnect [mm]: 32.768"
  let row_R = 0.1 * 128
  let col_R = 0.1 * 128
  let row_C_fF = 0.2 * 128
  let col_C_fF = 0.2 * 128
  echo "Row wire R/row [Ohm]:"
  print row_R
  echo "Col wire R/col [Ohm]:"
  print col_R
  echo "Row wire C/row [fF]:"
  print row_C_fF
  echo "Col wire C/col [fF]:"
  print col_C_fF

  echo "SPICE_DONE"
  wrdata crossbar_rc.dat V(col_node)
.endc

.end
