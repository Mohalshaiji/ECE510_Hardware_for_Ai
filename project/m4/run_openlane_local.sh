#!/usr/bin/env bash
# =============================================================================
# run_openlane_local.sh -- RC-DFA M4 local synthesis script
# ECE 510 Spring 2026
#
# Usage: cd <repo_root> && ./project/m4/run_openlane_local.sh
#
# What this does:
#   Step 1  Compile and run M4 simulation (iverilog) -- confirms PASS before synth
#   Step 2  Write pnr_constraints.sdc (clock + analog arrival from M3 ngspice)
#   Step 3  Run OpenLane 2 PnR with MAX_FANOUT_CONSTRAINT=8 (Task 3)
#   Step 4  Copy timing / area / power reports into project/m4/synth/
#
# Prerequisites (same as M3):
#   - openlane_env virtualenv at /mnt/c/Users/mohal/Downloads/openlane_env
#   - sky130A PDK under $PDK_ROOT (default ~/.volare)
#   - iverilog 12 on PATH
#   - OpenLane 2.3.7: pip install openlane==2.3.7
#
# After this script completes, commit everything in project/m4/ and tag:
#   git add project/m4/
#   git commit -m "M4: synthesis results + updated reports"
#   git tag m4-submission
#   git push origin main m4-submission
# =============================================================================

set -euo pipefail

source /mnt/c/Users/mohal/Downloads/openlane_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
M4="$SCRIPT_DIR"
REPO_ROOT="$(cd "$M4/../../.." && pwd)"

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'
info() { echo -e "${GRN}[m4]${RST} $*"; }
warn() { echo -e "${YLW}[warn]${RST} $*"; }
die()  { echo -e "${RED}[FAIL]${RST} $*"; exit 1; }

info "M4 synthesis script starting"
info "Repo root : $REPO_ROOT"
info "M4 dir    : $M4"

# =============================================================================
# Step 1: Simulation -- confirm PASS before spending time on synthesis
# =============================================================================
info "=== Step 1: M4 simulation (iverilog) ==="
cd "$M4"
mkdir -p sim

if ! command -v iverilog &>/dev/null; then
    warn "iverilog not found -- skipping simulation check (install with: sudo apt install iverilog)"
else
    iverilog -g2012 -I tb/ \
        -o sim/tb_top.vvp \
        tb/tb_top.sv \
        rtl/top.sv \
        rtl/interface.sv \
        rtl/compute_core.sv \
        2>&1 | tee sim/compile.log

    vvp sim/tb_top.vvp 2>&1 | tee sim/final_run.log

    if grep -q "RESULT: PASS" sim/final_run.log; then
        info "Simulation: RESULT: PASS"
    else
        die "Simulation FAILED -- fix RTL before synthesizing. See sim/final_run.log"
    fi
fi

# =============================================================================
# Step 2: Write pnr_constraints.sdc
#   Reuses the analog arrival time from M3 ngspice (t_99 = 33.380 ps).
#   If you re-ran ngspice for M4, replace the set_input_delay value below
#   with the updated t_99 from project/m3/synth/analog_arrival.tcl.
# =============================================================================
info "=== Step 2: Writing pnr_constraints.sdc ==="
mkdir -p "$M4/synth"

cat > "$M4/synth/pnr_constraints.sdc" << 'SDC'
# pnr_constraints.sdc -- clock and IO constraints for OpenROAD PnR
# RC-DFA M4 | ECE 510 Spring 2026
# Analog arrival time reused from M3 ngspice characterisation (t_99 = 33.380 ps).
create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay  0.5 -clock clk [get_ports {rst_n s_axis_tdata s_axis_tvalid s_axis_tlast s_axis_tkeep m_axis_tready}]
set_output_delay 0.5 -clock clk [get_ports {s_axis_tready m_axis_tdata m_axis_tvalid m_axis_tlast m_axis_tkeep sample_pulse}]
set_input_delay 0.033380 -clock clk [get_ports {adc_out}]
SDC

info "pnr_constraints.sdc written"

# =============================================================================
# Step 3: OpenLane 2 PnR
# =============================================================================
info "=== Step 3: OpenLane 2 PnR (MAX_FANOUT_CONSTRAINT=8) ==="

export PDK_ROOT="${PDK_ROOT:-$HOME/.volare}"
info "PDK_ROOT = $PDK_ROOT"

PDK_HASH="bdc9412b3e468c102d01b7cf6337be06ec6e9c9a"
info "Ensuring sky130A PDK is enabled (hash $PDK_HASH)..."
volare fetch --pdk sky130 "$PDK_HASH" 2>/dev/null || true
volare enable --pdk sky130 "$PDK_HASH" 2>/dev/null || true

# config.json is already written at project/m4/synth/config.json with:
#   MAX_FANOUT_CONSTRAINT: 8   (Task 3 -- was 16 in M3)
#   VERILOG_FILES pointing to m4/rtl/interface.sv (Task 1+2)
#   VERILOG_FILES pointing to m4/rtl/compute_core.sv (Task 3 start_r)
info "Using synth/config.json:"
cat "$M4/synth/config.json"
echo ""

cd "$M4/synth"

sudo nix --extra-experimental-features 'nix-command flakes' \
    run github:efabless/openlane2 -- --pdk-root "$PDK_ROOT" config.json \
    2>&1 | tee openlane_run.log

# =============================================================================
# Step 4: Copy reports out of the run directory
# =============================================================================
info "=== Step 4: Copying reports ==="

RUN_DIR=$(ls -td runs/RUN_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    die "No OpenLane run directory found under synth/runs/ -- check openlane_run.log"
fi
info "OpenLane run directory: $RUN_DIR"

# ── Timing report (post-route STA) ──────────────────────────────────────────
TIMING_SRC=$(find "$RUN_DIR" -path "*/stapostpnr*" -name "*.rpt" \
             | grep -i "timing\|setup\|sta" | head -1)
if [ -n "$TIMING_SRC" ]; then
    cp "$TIMING_SRC" "$M4/synth/timing_report.txt"
    info "timing_report.txt written from $TIMING_SRC"
else
    # Fallback: find any timing rpt
    TIMING_SRC=$(find "$RUN_DIR" -name "*.rpt" | grep -i "timing" | tail -1)
    [ -n "$TIMING_SRC" ] && cp "$TIMING_SRC" "$M4/synth/timing_report.txt" \
        && info "timing_report.txt written (fallback)" \
        || warn "Could not find timing report -- check $RUN_DIR manually"
fi

# ── Area report (cell frequency / stat) ─────────────────────────────────────
AREA_SRC=$(find "$RUN_DIR" -name "*.rpt" | grep -iE "cellfrequency|stat|area" | head -1)
if [ -n "$AREA_SRC" ]; then
    cp "$AREA_SRC" "$M4/synth/area_report.txt"
    info "area_report.txt written from $AREA_SRC"
else
    warn "Could not find area report -- check $RUN_DIR manually"
fi

# ── Power report ─────────────────────────────────────────────────────────────
POWER_SRC=$(find "$RUN_DIR" -name "*.rpt" | grep -i "power" | tail -1)
if [ -n "$POWER_SRC" ]; then
    cp "$POWER_SRC" "$M4/synth/power_report.txt"
    info "power_report.txt written from $POWER_SRC"
else
    warn "Could not find power report -- check $RUN_DIR manually"
fi

# ── openlane_pnr.log (step-level progress) ───────────────────────────────────
PNR_LOG=$(find "$RUN_DIR" -name "*.log" | head -1)
[ -n "$PNR_LOG" ] && cp "$PNR_LOG" "$M4/synth/openlane_pnr.log" \
    && info "openlane_pnr.log written" || true

info "=== All steps complete ==="
echo ""
echo "Files produced / updated in project/m4/:"
echo "  sim/final_run.log          (iverilog: RESULT: PASS)"
echo "  synth/pnr_constraints.sdc  (clock + analog arrival constraints)"
echo "  synth/config.json          (MAX_FANOUT_CONSTRAINT=8, M4 RTL paths)"
echo "  synth/openlane_run.log     (full OpenLane 2 stdout)"
echo "  synth/openlane_pnr.log     (step-level progress)"
echo "  synth/timing_report.txt    (post-route STA: WNS, TNS, critical path)"
echo "  synth/area_report.txt      (cell counts, total area)"
echo "  synth/power_report.txt     (sequential / combinational / clock / total)"
echo ""
echo "Next steps:"
echo "  1. Review synth/timing_report.txt -- confirm WNS >= 0 at nom_tt_025C_1v80"
echo "  2. Update project/m4/bench/benchmark.md power figures with new power_report.txt values"
echo "  3. git add project/m4/"
echo "  4. git commit -m 'M4: synthesis with MAX_FANOUT_CONSTRAINT=8, Task 1/2/3 RTL'"
echo "  5. git tag m4-submission && git push origin main m4-submission"
