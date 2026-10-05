#!/usr/bin/env bash
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
export OLLAMA_NUM_PARALLEL=1
export LLAMA_ARG_CACHE_RAM=0
exec ollama serve
