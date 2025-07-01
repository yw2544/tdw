#!/usr/bin/env bash
set -euo pipefail

# 1) task index
idx=$1
# 2) calculate port & output directory
port=$((1071 + idx))
out=/workspace/output/run_$(printf "%02d" "$idx")

echo "[$(date +%T)] start task #$idx → $out (port $port)"

# 3) ensure output directory exists
mkdir -p "$out"

# 4) export DISPLAY for Python/Unity headless build
export DISPLAY=${DISPLAY:-:1}

# 5) run Python pipeline (background), Python will launch TDW.x86_64
python /workspace/pipeline.py \
    --output "$out" \
    --port   "$port" \
    --objects 5 --room 10 10 \
    --seed   $((42 + idx)) \
    --cols   4 --thumb 512 &

pid=$!
echo "[$(date +%T)] Python PID=$pid, waiting for finish..."

# 6) wait for Python pipeline to finish
wait $pid

echo "[$(date +%T)] task #$idx finished → $out"
