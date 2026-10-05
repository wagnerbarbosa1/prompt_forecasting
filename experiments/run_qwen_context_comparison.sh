#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_ENVS="${CONDA_ENVS:-/home/wagner-barbosa/miniconda3/envs}"
OUTPUT=experiments/qwen_context_comparison
mkdir -p "$OUTPUT/logs"
for CONTEXT_POLICY in 15_days expanding; do
    "$CONDA_ENVS/promptcast/bin/python" -m src.pipelines.forecasting_pipeline \
        --model qwen --config "configs/qwen_context_${CONTEXT_POLICY}.json" \
        --output "$OUTPUT/$CONTEXT_POLICY" > "$OUTPUT/logs/${CONTEXT_POLICY}.log" 2>&1
done
"$CONDA_ENVS/promptcast/bin/python" -m src.pipelines.compare_qwen_contexts --output "$OUTPUT"
