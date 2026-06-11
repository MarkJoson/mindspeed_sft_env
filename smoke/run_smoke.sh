#!/bin/bash
# 8-card SFT smoke test — run INSIDE the qwen36sft container (no docker wrapper).
# Random-init dense Qwen3.6 (8 layers: 6 linear_attention + 2 full_attention), NO weights.
# Goal: confirm the env runs end-to-end on 8x910B (FSDP2 + GDN ascendc + full-attn).
#
# cfg_smoke.yaml expects, inside the container:
#   - model/tokenizer dir at /workspace/qwen36-mini  (mini config, for tokenizer + arch)
#   - dataset at /workspace/smoke/data/mini_sft.json  (regenerate via gen_data.py)
# adjust those two paths in cfg_smoke.yaml to match your container, or symlink them.
set -ex
HERE="$(cd "$(dirname "$0")" && pwd)"
MM_DIR="${MM_DIR:-/workspace/MindSpeed-MM}"

export HF_HOME=/tmp/hf HF_DATASETS_CACHE=/tmp/hfds TRITON_CACHE_DIR=/tmp/tc
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"   # AscendC GDN NaN fix
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH}

cd "$MM_DIR"
python -m torch.distributed.run --nproc_per_node=8 --master_port=29555 \
  mindspeed_mm/fsdp/train/trainer.py "$HERE/cfg_smoke.yaml"
echo "SMOKE_EXIT=$?"
