#!/usr/bin/env bash
# Use existing conda environments; never install or update packages here.
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_ENVS="${CONDA_ENVS:-/home/wagner-barbosa/miniconda3/envs}"
mkdir -p experiments/benchmark/logs
"$CONDA_ENVS/promptcast/bin/python" -m src.pipelines.forecasting_pipeline --model qwen > experiments/benchmark/logs/qwen.log 2>&1
ollama stop qwen3:1.7b
HF_HUB_OFFLINE=1 "$CONDA_ENVS/chronos/bin/python" -m src.pipelines.forecasting_pipeline --model chronos > experiments/benchmark/logs/chronos.log 2>&1
HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false "$CONDA_ENVS/timesfm/bin/python" -m src.pipelines.forecasting_pipeline --model timesfm > experiments/benchmark/logs/timesfm.log 2>&1
"$CONDA_ENVS/promptcast/bin/python" -m src.pipelines.compare_models
