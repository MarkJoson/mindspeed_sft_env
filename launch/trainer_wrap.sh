#!/bin/bash
# per-rank triton cache (16 procs on one filesystem must not race a shared cache
# dir during first-run compile), then exec the trainer. used via torchrun --no-python.
export TRITON_CACHE_DIR="${TRITON_CACHE_BASE:-/tmp/tc}/rank${RANK}"
mkdir -p "$TRITON_CACHE_DIR"
exec python3 -u "${MM_DIR:-/workspace/MindSpeed-MM}/mindspeed_mm/fsdp/train/trainer.py" "$@"
