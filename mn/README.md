# mn/ — 训练配置参考（16 卡 MoE）

Qwen3.6-35B-A3B MoE 的一组**已跑通**的参考配置。这是从实验集里挑出的代表性子集（省略了
U-scaling 调参时的一堆实验变体，如 `u8ab/u8clean/u8nt/u4ns/...`）。

> ⚠️ 所有 `model_name_or_path` / `dataset` / `load` / `save` 路径已脱敏为 `/path/to/...`，
> 用前改成你自己的模型/数据/输出路径。MoE 的 `load:` 要指向 **DCP** 目录（见教程 §2）。

| 配置 | 场景 |
|---|---|
| `cfg_moe_30k.yaml` | 30k 序列，U=1 / dp16，grad_accum4→bs64，ascendc GDN ——**生产基准**（教程 §4 逐字段拆它） |
| `cfg_moe_32k_bs64.yaml` | 32k 序列，U=1，bs64 |
| `cfg_moe_128k_asc.yaml` | 128k 长上下文，ascendc GDN |
| `cfg_moe_256k_u16a.yaml` | 256k，Ulysses **U=16**（ascendc），长上下文基准 |
| `cfg_moe_256k_u8tri.yaml` | 256k，Ulysses **U=8**（triton GDN），dp=2 |
| `cfg_moe_256k_prof.yaml` | 256k + 打开 profiler 的示例（`tools.profile.enable: true`） |

最小可跑的随机初始化 dense 配置见 `../smoke/cfg_smoke.yaml`。

## 跑法
- 双机 16 卡：见教程 §5B（`run_moe.sh <node_rank> <cfg> [timeout]`）。
- 单机 16 卡：`../launch/run_16c.sh mn/cfg_moe_30k.yaml`（见教程 §5C）。

> 长序列 GDN 后端：256k+CP 用 `triton`（`u8tri`），常规 ≤32k/卡 U=1 用 `ascendc`。原因见教程 §4 model 块。
