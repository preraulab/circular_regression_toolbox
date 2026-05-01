#!/bin/bash
# Drive the 5-way circular-regression comparison on a chosen dump.
# Usage: run_compare.sh <dump_file> [results_dir]

DUMP=${1:?usage: run_compare.sh <dump_file> [results_dir]}
RESULTS_DIR=${2:-/Users/Mike/code/projects/trends_v_individual/stats/circular_regression/results/$(basename "${DUMP%.mat}")}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$RESULTS_DIR"

echo "=== $DUMP -> $RESULTS_DIR ==="

/Applications/MATLAB_R2025a.app/bin/matlab -batch \
  "addpath('$SCRIPT_DIR'); export_for_compare('$DUMP','$RESULTS_DIR')" 2>&1 | tail -3

Rscript "$SCRIPT_DIR/compare_brms.R"   "$RESULTS_DIR" 2>&1 | tail -3
Rscript "$SCRIPT_DIR/compare_lme4.R"   "$RESULTS_DIR" 2>&1 | tail -3
Rscript "$SCRIPT_DIR/compare_bpnreg.R" "$RESULTS_DIR" 2>&1 | tail -3

/Applications/MATLAB_R2025a.app/bin/matlab -batch \
  "addpath('$SCRIPT_DIR'); plot_5way_compare('$DUMP','$RESULTS_DIR')" 2>&1 | tail -15

OUT_PNG="$RESULTS_DIR/five_way_compare.png"
[ -f "$OUT_PNG" ] && echo "Wrote $OUT_PNG"
