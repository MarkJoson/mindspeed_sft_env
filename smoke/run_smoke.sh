#!/bin/bash
# 8-card SFT smoke test inside the qwen36sft image.
set -x
export DOCKER_API_VERSION="$(docker version --format '{{.Server.APIVersion}}')"
DEV=""
for i in 0 1 2 3 4 5 6 7; do DEV="$DEV --device=/dev/davinci$i"; done
docker run --rm --name qwen36sft_smoke \
  --net=host --ipc=host --shm-size=64g \
  $DEV \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v $HOME/qwen36-mini:/workspace/qwen36-mini:ro \
  -v $HOME/qwen36-smoke:/workspace/smoke \
  -e HCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  qwen36sft:latest \
  bash -lc 'cd /workspace/MindSpeed-MM && \
    export HF_HOME=/tmp/hf HF_DATASETS_CACHE=/tmp/hfds TRITON_CACHE_DIR=/tmp/tc && \
    python -m torch.distributed.run --nproc_per_node=8 --master_port=29555 \
      mindspeed_mm/fsdp/train/trainer.py /workspace/smoke/cfg_smoke.yaml'
echo "SMOKE_EXIT=$?"
