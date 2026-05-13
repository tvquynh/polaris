#!/bin/bash
# run_phase4b_optuna100.sh — Phase 4b winner (FTE+LS+Optuna-100T) × 10 seeds.
# Same lock convention as Phase 4a.
set -e

PYTHON_BIN="${PYTHON_BIN:-/path/to/venv/bin/python}"
PARQUET_DIR="${PARQUET_DIR:-/path/to/ember2024_parquet}"
RELABEL_FRAME="${RELABEL_FRAME:-/path/to/relabel_frame.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CODE_DIR="${CODE_DIR:-.}"
NUM_THREADS="${NUM_THREADS:-60}"

CFG="fte_ls_optuna100"
SEEDS="${SEEDS:-42 123 456 789 1011 2026 3141 4242 5555 6789}"

echo "=== Phase 4b START $(date) ==="
for S in $SEEDS; do
    RES_DIR="$OUTPUT_DIR/$CFG/seed_$S"
    mkdir -p "$RES_DIR"

    [ -f "$RES_DIR/metrics.json" ] && { echo "[SKIP DONE] seed_$S"; continue; }
    mkdir "$RES_DIR/RUNNING.lock" 2>/dev/null || { echo "[SKIP LOCKED] seed_$S"; continue; }
    echo "PID=$$ start=$(date -Iseconds)" > "$RES_DIR/RUNNING.lock/info.txt"

    "$PYTHON_BIN" "$CODE_DIR/pipeline_v2.py" \
        --config_id "$CFG" --seed "$S" \
        --parquet_dir "$PARQUET_DIR" \
        --relabel_frame "$RELABEL_FRAME" \
        --output_dir "$OUTPUT_DIR" \
        --num_threads "$NUM_THREADS" \
        >> "$RES_DIR/run.log" 2>&1
    rc=$?

    [ $rc -ne 0 ] && touch "$RES_DIR/FAILED.flag"
    rm -rf "$RES_DIR/RUNNING.lock"
    echo "[END rc=$rc] seed_$S at $(date +%H:%M:%S)"
done
echo "=== Phase 4b END $(date) ==="
