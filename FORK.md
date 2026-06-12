# Fork 工作流 —— `MarkJoson/MindSpeed-MM`

镜像不再走 "upstream + patch",而是**直接 clone fork 的修改分支**。本文件记录怎么把
`patches/mindspeed-mm-qwen36-sft.patch` 落成 fork 上的一个分支。

- **上游基点**:`Ascend/MindSpeed-MM` @ `cd34547968d9de915b826f1cba9e05d27153167b`
- **分支名**:`qwen36-sft`(与 `docker/Dockerfile` 的 `ARG MINDSPEED_MM_REF` 一致)
- **改动**:`patches/mindspeed-mm-qwen36-sft.patch`(14 文件,+254/-42;纯代码,已脱敏)

## 一次性:建立 fork 分支

```bash
# 1) 在 GitHub 把 Ascend/MindSpeed-MM fork 成 MarkJoson/MindSpeed-MM

# 2) 基于 pin 的上游 commit 建分支并打补丁
git clone https://github.com/MarkJoson/MindSpeed-MM.git
cd MindSpeed-MM
git remote add upstream https://github.com/Ascend/MindSpeed-MM.git
git fetch upstream cd34547968d9de915b826f1cba9e05d27153167b
git checkout cd34547968d9de915b826f1cba9e05d27153167b -b qwen36-sft
git apply --whitespace=nowarn /path/to/mindspeed_sft_env/patches/mindspeed-mm-qwen36-sft.patch
git add -A
git commit -m "Qwen3.6 SFT: GDN single-step opt + MoE create_causal_mask fix + fused CE"
git push -u origin qwen36-sft
```

之后镜像构建直接 `docker buildx build docker/`(Dockerfile 默认 clone `MarkJoson/MindSpeed-MM`
的 `qwen36-sft` 分支,无需再打补丁)。要换分支名就改 `--build-arg MINDSPEED_MM_REF=...`。

## 补丁里有什么(想拆成多个 commit 的话,按组拆)

| 组 | 文件 | 作用 |
|---|---|---|
| **GDN 单步优化** | `ops/gdn/triton/solve_tril.py`、`triton/utils.py`、`ops/gdn/flash_*_gated_delta_rule.py`、`ops/flash_attn/flash_attn.py`、`ops/fully_shard/fully_shard.py`、`tools/profiler.py`、`train/train_engine.py`、`train/trainer.py` | `solve_tril` 的 `NT` 改运行时参数(去掉每步重编译)、`prepare_chunk_indices` 向量化、`cu_seqlens` 缓存 + profiler/step 埋点。~1.8× MFU |
| **transformers 兼容修复** | `models/qwen3_5/modeling_qwen3_5.py`(dense)、`models/qwen3_5_moe/modeling_qwen3_5_moe.py`(MoE) | 去掉 `create_causal_mask` 的 `cache_position` 入参。**MoE 这个是旧 docker patch 缺的关键** —— 不补,35B-A3B 跑不起来 |
| **fused CE** | `loss/loss_func.py` + `features/memory/chunkloss/fused_lm_ce.py` + `fused_lm_ce_maxsum.py` | 分块 LM head loss 的融合实现(vocab-block grad_weight),省显存/提速。仅优化,非正确性 |

> 注:`docstring-parser==0.18.0` 是 `mm-convert`(HF→DCP)需要的依赖,已加进 `docker/requirements.lock.txt`(不是 MM 代码,不在本补丁里)。

## 维护

- 上游升级后:`git checkout <new_upstream> -b qwen36-sft-<date>`,重新 `git apply` 本补丁,解冲突,push,改 `MINDSPEED_MM_REF`。
- 补丁是从集群 working tree 生成的快照(`git diff HEAD` + intent-to-add 的两个新文件);要再生成,在集群 `MindSpeed-MM` 里 `git --no-pager diff HEAD`(记得先 `git add -N` 那两个 `fused_lm_ce*.py`)。
