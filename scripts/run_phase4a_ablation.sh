#!/bin/bash
# run_phase4a_ablation.sh — no-HPO ablation across 15 configurations × 10 seeds.
# Adapt the paths below to your environment.
set -e

# ── Configuration (edit these for your environment) ─────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/path/to/venv/bin/python}"
PARQUET_DIR="${PARQUET_DIR:-/path/to/ember2024_parquet}"
RELABEL_FRAME="${RELABEL_FRAME:-/path/to/relabel_frame.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CODE_DIR="${CODE_DIR:-.}"
NUM_THREADS="${NUM_THREADS:-60}"

CONFIGS="baseline_B fte fte_mgep_top3 fte_mgep_top5 fte_sar fte_ls fte_mgep_sar fte_mgep_sar_ls fte_mgep_sar_ls_peto_noHPO fte_beh_only R_5 R_20"
SEEDS="42 123 456 789 1011 2026 3141 4242 5555 6789"

echo "=== Phase 4a START $(date) ==="
for CFG in $CONFIGS; do
    for S in $SEEDS; do
        RES_DIR="$OUTPUT_DIR/$CFG/seed_$S"
        mkdir -p "$RES_DIR"

        [ -f "$RES_DIR/metrics.json" ] && { echo "[SKIP DONE] $CFG/seed_$S"; continue; }
        mkdir "$RES_DIR/RUNNING.lock" 2>/dev/null || { echo "[SKIP LOCKED] $CFG/seed_$S"; continue; }

        echo "[START] $CFG seed_$S at $(date +%H:%M:%S)"
        echo "PID=$$ start=$(date -Iseconds)" > "$RES_DIR/RUNNING.lock/info.txt"

        "$PYTHON_BIN" "$CODE_DIR/pipeline_v2.py" \
            --config_id "$CFG" --seed "$S" \
            --parquet_dir "$PARQUET_DIR" \
            --relabel_frame "$RELABEL_FRAME" \
            --output_dir "$OUTPUT_DIR" \
            --num_threads "$NUM_THREADS" \
            --no_hpo \
            >> "$RES_DIR/run.log" 2>&1
        rc=$?

        [ $rc -ne 0 ] && touch "$RES_DIR/FAILED.flag"
        rm -rf "$RES_DIR/RUNNING.lock"
        echo "[END rc=$rc] $CFG seed_$S at $(date +%H:%M:%S)"
    done
done
echo "=== Phase 4a END $(date) ==="
