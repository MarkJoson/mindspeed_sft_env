# MindSpeed-MM 拉起 SFT 教程（Qwen3.6 / Ascend 910B）

本教程基于本项目已**验证可跑**的环境（`ssh npu`，2 节点 16 卡 910B3）整理，命令均来自实跑的
`Dockerfile` / `run_moe.sh` / `run_smoke.sh` / `cfg_*.yaml`，不是通用模板。

> 一句话原理：MindSpeed-MM 的 SFT 是 **FSDP2** 驱动，入口就一个
> `mindspeed_mm/fsdp/train/trainer.py <config.yaml>`，**一份 YAML 管全部**
> （并行 / 数据 / 模型 / features / 训练 / 工具）。

---

## 0. 两条模型路线

| 模型 | model_id | 权重加载 | 备注 |
|---|---|---|---|
| Qwen3.6-27B（dense） | `qwen3_5` | HF 目录直接加载 | 入门用这个 |
| Qwen3.6-35B-A3B（MoE） | `qwen3_5_moe` | **必须先转 DCP** | HF-direct 装不进 MoE experts |

---

## 1. 环境

### 软件栈（已验证的精确版本）

| 组件 | 来源 | commit / 版本 |
|---|---|---|
| Base | CANN 9.0.0 aarch64 · torch 2.10.0+cpu · torch_npu 2.10.0 · triton-ascend 3.2.1 |
| transformers | github huggingface/transformers | `94246e68`（含 Qwen3.6 支持） |
| MindSpeed | gitcode Ascend/MindSpeed | `5753d412` |
| MindSpeed-MM | github Ascend/MindSpeed-MM | `cd345479` **+ 本项目 patch** |
| flash-linear-attention-npu | flashserve/flash-linear-attention-npu | `eabe36b`（GDN AscendC 算子） |

### 方式 A：用已构建的 Docker 镜像（推荐）

镜像 `qwen36sft:latest` 已构建验证。构建脚本见 `qwen36sft_build/dev-sft/Dockerfile`
（BuildKit）。直接 `docker run` 即可，见第 5 节。

### 方式 B：裸机 venv（集群上的 `~/qwen36-sft-env` 就是这么装的）

按 Dockerfile 的顺序，**全程 `--no-deps`**（依赖闭包已被 lock 锁死，避免 pip 乱解析）：

```bash
python3.11 -m venv ~/qwen36-sft-env && source ~/qwen36-sft-env/bin/activate

# 1) 锁定的依赖闭包
pip install --no-deps -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple

# 2) transformers @ pinned commit
pip install --no-deps --no-build-isolation \
  "transformers @ git+https://github.com/huggingface/transformers.git@94246e689c03551bd97624d436de4c7e2d937063"

# 3) MindSpeed（editable）
git clone https://gitcode.com/Ascend/MindSpeed.git && \
  git -C MindSpeed checkout --detach 5753d412882317b113e08103a2b0b7155345096a && \
  pip install -e MindSpeed --no-deps

# 4) MindSpeed-MM（editable）+ 本项目 patch
git clone https://github.com/Ascend/MindSpeed-MM.git && \
  git -C MindSpeed-MM checkout --detach cd34547968d9de915b826f1cba9e05d27153167b && \
  git -C MindSpeed-MM apply --whitespace=nowarn mindspeed_mm_qwen36_sft.patch && \
  pip install -e MindSpeed-MM --no-deps

# 5) fla_npu（GDN AscendC 算子）：用预编译产物
bash fla-npu-fla_npu_linux-aarch64.run --quiet --install-for-all
pip install --no-deps --force-reinstall fla_npu-1.0.0-cp311-cp311-linux_aarch64.whl
```

### ⚠️ 关键运行时环境变量（不设会出错，不是可选）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export TASK_QUEUE_ENABLE=0          # ★ AscendC GDN 的 NaN 修复（异步 task-queue 竞态）。不设 → GDN 出 NaN
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib:$LD_LIBRARY_PATH  # ★ AscendC 算子加载
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 多机还要：
export HCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0   # 改成你的网卡名
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
```

---

## 2. 权重准备

### dense（27B）
HF 目录直接用：`model_name_or_path` 与 `training.load` 都指向 HF 权重目录即可。

### MoE（35B-A3B）——必须转 DCP
`HF-direct 加载不了 MoE experts`，要先用 MindSpeed-MM 自带转换器把 HF → DCP（experts 会被转置）：

```bash
cd MindSpeed-MM
# 入口：checkpoint/convert_cli.py（底层 checkpoint/common/hf_to_dcp.py）
python checkpoint/convert_cli.py --help     # 先看确切参数
# 形如：hf_to_dcp，输入 HF 目录，输出 <model>-dcp 目录
#   → 产出 /path/Qwen3.6-35B-A3B-dcp
```

训练 config 里这样指：
```yaml
training:
  load: /path/Qwen3.6-35B-A3B-dcp     # 指向 DCP，不是 HF
  load_format: auto                    # 顶层无 safetensors → 自动识别 dcp
  load_rank0_and_broadcast: false      # FSDP 原生 sharded DCP 加载
```

> 训练完要导回 HF：用 `checkpoint/common/merge_dcp_to_hf.py`。

---

## 3. 数据准备

格式：一个 JSON 数组，每条是 HF messages 格式（纯文本 SFT 时 `images: []`）：

```json
[
  {"messages": [
      {"role": "user", "content": "Explain how a neural network learns."},
      {"role": "assistant", "content": "A neural network learns by ..."}
    ],
   "images": []}
]
```

最小例子见 `qwen36sft_build/smoke/gen_data.py`（生成 96 条）。

> ⚠️ **样本数必须 ≥ global_batch**（= `world_size × micro_batch_size × grad_accum`），
> 否则 `BaseRandomBatchSampler` 会除零崩溃。

---

## 4. 配置文件逐块讲解（以 `cfg_moe_30k.yaml` 为基准）

### `parallel` — 并行/分片
```yaml
parallel:
  fully_shard_parallel_size: auto      # auto → world_size（16）。FSDP2 全分片
  fsdp_plan:
    apply_modules:                     # 分片+prefetch 单元；顺序影响 prefetch，按官方 MoE 顺序
      - model.visual
      - model.visual.blocks.{*}
      - model.language_model
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}
      - model.language_model.layers.{*}.linear_attn
      - model.language_model.layers.{*}.mlp.experts
      - lm_head
    param_dtype: bf16
    reduce_dtype: fp32                  # 梯度规约用 fp32，数值更稳
    cpu_offload: false
  ulysses_parallel_size: 1             # 长序列才开 CP（>32k/卡）；开了 dp 会降，按需
  expert_parallel_size: 1              # MoE 专家并行；1=experts 走 FSDP 分片（官方默认）
  ep_plan:
    apply_modules: [model.language_model.layers.{*}.mlp.experts]
```

### `model`
```yaml
model:
  model_id: qwen3_5_moe                # dense 用 qwen3_5
  model_name_or_path: <HF 路径>
  attn_implementation: flash_attention_2   # 开 CP 时必须是这个
  freeze: [model.visual]              # 纯语言 SFT 冻结视觉塔
  gdn_implementation: ascendc         # GDN 后端：ascendc（需 TASK_QUEUE_ENABLE=0）/ triton
  use_grouped_expert_matmul: true     # MoE：256 experts 的 GMM 融合
  skip_gdn_recompute: true            # 把 GDN 激活 offload 到 CPU（替代重算）
  skip_flash_attn_recompute: true     # 把 full-attn 激活 offload 到 CPU
```
> GDN 后端选择（实测要点）：
> - `ascendc`：cube 利用率高，**但 >16k/卡 + CP 场景会 triton 编译 runaway**；
>   常规 ≤32k/卡 U=1 跑得很好。
> - `triton`：到处能跑但 GDN kernel 偏 vector-core、cube 利用低（~18%）。
> 长序列 + CP（如 256K U=8）目前只能用 `triton`。

### `features`
```yaml
features:
  recompute: true                     # 梯度检查点（35B 必开，否则放不下）
  recompute_plan:
    apply_modules: [model.visual.blocks.{*}, model.language_model.layers.{*}]
  enable_chunk_loss: true             # 分块 lm_head loss，省词表 logits 显存
  chunkloss_plan: {apply_module: lm_head, chunk_size: 1024}
  enable_activation_offload: false    # 通用激活 offload（与 skip_* 区别开）
```

### `training`
```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 4      # global batch = world(16) × mbs(1) × ga(4) = 64
  lr: 1.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  train_iters: 3
  init_model_with_meta_device: true   # meta device 初始化再加载，省峰值内存
  optimizer: adamw
  adam_fused: true
  load: <DCP 路径>                    # MoE 用 DCP；dense 用 HF 路径
  load_format: auto
  load_rank0_and_broadcast: false
  save: <输出目录>
  plugin:                             # ★ 注册模型与数据插件，必填
    - mindspeed_mm/fsdp/models/qwen3_5_moe
    - mindspeed_mm/fsdp/data/datasets/huggingface
```
> 随机初始化跑 smoke（不加载权重）：省略 `load`（→ None）+ `init_model_with_meta_device: true`
> + `load_rank0_and_broadcast: false`，见 `cfg_smoke.yaml`。

### `tools`（按需开 profiler）
```yaml
tools:
  profile:
    enable: false                     # 要采就 true
    profile_type: static
    ranks: [0]
    static_param: {level: level1, start_step: 9, end_step: 11, with_cpu: true,
                   save_path: /tmp/prof_moe, aic_metrics_type: PipeUtilization}
```

---

## 5. 启动

### A. 单机 8 卡（smoke / 开发）

裸机：
```bash
cd MindSpeed-MM
export HF_HOME=/tmp/hf TRITON_CACHE_DIR=/tmp/tc
python -m torch.distributed.run --nproc_per_node=8 --master_port=29555 \
  mindspeed_mm/fsdp/train/trainer.py /path/cfg_smoke.yaml
```

Docker（即 `run_smoke.sh`，挂 8 张卡 + 驱动 + 数据/权重目录）：
```bash
docker run --rm --net=host --ipc=host --shm-size=64g \
  $(for i in 0 1 2 3 4 5 6 7; do echo --device=/dev/davinci$i; done) \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v <权重目录>:/workspace/qwen36-mini:ro -v <smoke目录>:/workspace/smoke \
  -e HCCL_SOCKET_IFNAME=<网卡> -e GLOO_SOCKET_IFNAME=<网卡> \
  qwen36sft:latest \
  bash -lc 'cd /workspace/MindSpeed-MM && \
    python -m torch.distributed.run --nproc_per_node=8 --master_port=29555 \
      mindspeed_mm/fsdp/train/trainer.py /workspace/smoke/cfg_smoke.yaml'
```

### B. 双机 16 卡（MoE 生产，即 `run_moe.sh`）

`run_moe.sh <node_rank> <cfg_name> [timeout_sec]` 关键内核：
```bash
python3 -m torch.distributed.run --nnodes=2 --node_rank=$NR --nproc_per_node=8 --no-python \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  trainer_wrap.sh mn/cfg_${CFG}.yaml
# trainer_wrap.sh 只做一件事：设 per-rank TRITON_CACHE_DIR 再 exec python trainer.py "$@"
```

两个节点分别执行（同一 MASTER_ADDR/PORT，node_rank 0 / 1）：
```bash
# 节点0（master）
MASTER_ADDR=10.0.4.98 MASTER_PORT=29761 bash run_moe.sh 0 moe_30k 1800
# 节点1
MASTER_ADDR=10.0.4.98 MASTER_PORT=29761 bash run_moe.sh 1 moe_30k 1800
```
> `TRITON_CACHE_BASE=/tmp/tc_<cfg>` **持久化复用，别每次 `rm -rf`**——首跑 triton 编译很慢
> （30k/卡 ~5min），删了等于每次重编译。

---

## 6. 验证训练正常

1. 看日志 `mn/log_<cfg>_r<NR>.txt` 出现：
   ```
   iteration  1/  3 | ... | elapsed time per iteration (ms): ... | loss: 4.87E+00
   ```
   loss 合理（loaded 权重）或 ~ln(vocab)≈12（random init）。
2. `npu-smi info` 看 16 卡 HBM 占用 + AI Core 利用率。
3. **首跑慢 ≠ hang**：triton 在编译，`top` 看到 procCPU 很高就是在编译，等它出 iter1。

---

## 7. 这个项目踩过的坑（按重要性）

| 现象 | 根因 / 解法 |
|---|---|
| GDN 输出 NaN | **`TASK_QUEUE_ENABLE=0` 必设**（AscendC 异步竞态） |
| MoE 加载失败 / 权重错乱 | **必须 HF→DCP 转换**，HF-direct 装不进 experts |
| `BaseRandomBatchSampler` 除零 | 样本数 < global_batch；样本数 ≥ `world×mbs×ga` |
| 长序列 ascendc GDN 卡死编译 | >16k/卡 + CP 时 ascendc triton-runaway，改 `gdn_implementation: triton` |
| 误判 hang | 首跑是 triton 编译；`TRITON_CACHE_BASE` 持久化复用 |
| global batch 对不上 | = `world_size × micro_batch_size × gradient_accumulation_steps` |
| 算子加载不到 | `LD_LIBRARY_PATH` 要含 `fla_npu_transformer/op_api/lib` |
| neat_packing 取舍 | `true` 省 padding 但 GDN 变长 cu_seqlens；`false` 定长、GDN shape 稳定 |

---

## 附：最小上手清单

1. 装环境（方式 A 镜像 / 方式 B venv）+ 设那几个 env（尤其 `TASK_QUEUE_ENABLE=0`）。
2. 备数据：JSON `[{"messages":[...],"images":[]}]`，条数 ≥ global_batch。
3. dense 直接用 HF 权重；MoE 先 `convert_cli.py` 转 DCP。
4. 抄 `cfg_smoke.yaml`（dense/random）或 `cfg_moe_30k.yaml`（MoE/加载）改路径。
5. 单机：`torchrun --nproc_per_node=8 trainer.py cfg.yaml`；双机：`run_moe.sh 0/1 <cfg>`。
6. 看 log 出 `iteration ... loss`，就算拉起来了。
</content>
</invoke>
