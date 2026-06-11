#!/bin/bash
# Single-node 16-card SFT launch (in-container, no docker wrapper).
# Usage: run_16c.sh <path/to/cfg.yaml> [timeout_sec]
#
# The config YAML is IDENTICAL to the 2-node setup — world_size is still 16
# (1 node x 16 procs), so `fully_shard_parallel_size: auto` -> 16 and dp/ulysses/ep
# are unchanged. Only the launch differs: one command, no node_rank / master_addr.
# Prereq: `npu-smi info` must show 16 NPUs on this host.
set -e
CFG="${1:?usage: run_16c.sh <cfg.yaml> [timeout_sec]}"; TMO="${2:-1800}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export MM_DIR="${MM_DIR:-/workspace/MindSpeed-MM}"

# validated runtime env (entrypoint sets these too; re-assert for a bare run)
export TASK_QUEUE_ENABLE=0 NON_MEGATRON=true MULTI_STREAM_MEMORY_REUSE=2 ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000 CPU_AFFINITY_CONF=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH}

# persistent per-cfg triton cache (don't rm between runs — first compile is slow)
export TRITON_CACHE_BASE="/tmp/tc_$(basename "$CFG" .yaml)"; mkdir -p "$TRITON_CACHE_BASE"

cd "$MM_DIR"
timeout "$TMO" python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=16 --master_port=29761 \
  --no-python "$HERE/trainer_wrap.sh" "$CFG"
