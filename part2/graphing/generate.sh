#!/bin/bash

rm -rf figures_breakdown figures_strong figures_weak tables

BACKEND="CSR_GPU"
SCALING_GHOST_MODE="alltoallv"      # choices: alltoallv, fullghost
DIAGNOSTIC_GHOST_MODE="alltoallv"   # used for runtime, communication, baseline tables

# must point to the root folder containing part2 with the results computed on disk
ROOT_DIR=$1

RESULT_DIR="$ROOT_DIR/merged_results"
INFO_DIR="$ROOT_DIR"

# must point to merged baseline csv
BASELINE_CSV=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMPUTE_CSVS=(
    "$RESULT_DIR/1_compute.csv"
    "$RESULT_DIR/2_compute.csv"
    "$RESULT_DIR/4_compute.csv"
)

PARSING_CSVS=(
    "$RESULT_DIR/1_parsing.csv"
    "$RESULT_DIR/2_parsing.csv"
    "$RESULT_DIR/4_parsing.csv"
)

MATRIX_INFO_CSVS=(
    "$INFO_DIR/real_matrices_info.csv"
    "$INFO_DIR/synthetic_matrices_info.csv"
)

REAL_INFO="$INFO_DIR/real_matrices_info.csv"
SYNTHETIC_INFO="$INFO_DIR/synthetic_matrices_info.csv"

OUT_STRONG="figures_strong"
OUT_WEAK="figures_weak"
OUT_BREAKDOWN="figures_breakdown"
OUT_TABLES="tables"

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

echo "Checking input files..."
for file in \
    "${COMPUTE_CSVS[@]}" \
    "${PARSING_CSVS[@]}" \
    "${MATRIX_INFO_CSVS[@]}" \
    "$BASELINE_CSV"
do
    check_file "$file"
done

echo "Removing previous outputs..."
rm -rf "$OUT_STRONG" "$OUT_WEAK" "$OUT_BREAKDOWN" "$OUT_TABLES"

echo "Creating output folders..."
mkdir -p "$OUT_STRONG" "$OUT_WEAK" "$OUT_BREAKDOWN" "$OUT_TABLES"

echo "Generating dataset tables..."
python3 "$SCRIPT_DIR/dataset.py" \
    --real "$REAL_INFO" \
    --synthetic "$SYNTHETIC_INFO" \
    --output "$OUT_TABLES"

echo "Generating strong-scaling figure..."
python3 "$SCRIPT_DIR/strong_scaling.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_STRONG" \
    --backend "$BACKEND" \
    --dataset real \
    --ghost-mode "$SCALING_GHOST_MODE" \
    --debug-per-matrix

echo "Generating weak-scaling figure..."
python3 "$SCRIPT_DIR/weak_scaling.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_WEAK" \
    --backend "$BACKEND" \
    --ghost-mode "$SCALING_GHOST_MODE" \
    --raw-time-debug

echo "Generating runtime-breakdown figure..."
python3 "$SCRIPT_DIR/runtime.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_BREAKDOWN" \
    --backend "$BACKEND" \
    --dataset real \
    --rank 4 \
    --ghost-mode "$DIAGNOSTIC_GHOST_MODE" \
    --sort ghost-fraction

echo "Generating compact CSR results table..."
python3 "$SCRIPT_DIR/table.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_TABLES" \
    --backend CSR_GPU CSR_CUSPARSE \
    --dataset real \
    --table-name csr_results_summary_table.tex

echo "Generating communication diagnostics table..."
python3 "$SCRIPT_DIR/communication.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --parsing "${PARSING_CSVS[@]}" \
    --matrix-info "${MATRIX_INFO_CSVS[@]}" \
    --output "$OUT_TABLES" \
    --backend "$BACKEND" \
    --dataset real \
    --rank 4 \
    --ghost-mode "$DIAGNOSTIC_GHOST_MODE" \
    --sort ghost-share \
    --table-name communication_diagnostics_table.tex

echo "Generating provided CPU baseline comparison table..."
python3 "$SCRIPT_DIR/baseline.py" \
    --baseline "$BASELINE_CSV" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_TABLES" \
    --backend CSR_GPU CSR_CUSPARSE \
    --dataset real \
    --ghost-mode "$DIAGNOSTIC_GHOST_MODE" \
    --table-name baseline_comparison_table.tex

echo "Generating CUDA-aware exchange comparison table..."
python3 "$SCRIPT_DIR/cuda_aware.py" \
    --compute "${COMPUTE_CSVS[@]}" \
    --output "$OUT_TABLES" \
    --host-backend CSR_CPU \
    --device-backend CSR_GPU \
    --dataset real \
    --rank 2 4 \
    --table-name cuda_aware_exchange_table.tex

echo "Done."
echo "Ghost mode for scaling figures:       $SCALING_GHOST_MODE"
echo "Ghost mode for diagnostic tables:     $DIAGNOSTIC_GHOST_MODE"
echo "Tables written to:                    $OUT_TABLES"
echo "Strong-scaling figures written to:    $OUT_STRONG"
echo "Weak-scaling figures written to:      $OUT_WEAK"
echo "Runtime-breakdown figures written to: $OUT_BREAKDOWN"
