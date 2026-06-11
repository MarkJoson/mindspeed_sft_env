# mindspeed_sft_env

Reproducible **MindSpeed-MM (FSDP2)** full-parameter **SFT** environment for **Qwen3.6**
(`Qwen3_5ForConditionalGeneration` / MoE `Qwen3_5MoeForConditionalGeneration`) on
**Ascend 910B**.

One YAML drives everything; the trainer entry is
`mindspeed_mm/fsdp/train/trainer.py <config.yaml>`.

## What's inside

| Path | What |
|---|---|
| `docs/mindspeed-mm-sft-tutorial.md` | **Start here** — end-to-end tutorial (env → weights → data → config → launch → verify → gotchas) |
| `docs/build-notes.md` | image build notes + the exact pinned stack |
| `docker/Dockerfile` | the validated image (CANN 9.0.0 / torch 2.10.0 / torch_npu 2.10.0 / triton-ascend 3.2.1 + MindSpeed-MM + fla_npu) |
| `docker/entrypoint.sh` | sources CANN, wires the fla_npu vendor lib, sets the validated runtime env |
| `docker/requirements.lock.txt` | 123-package pinned pip closure |
| `docker/patches/mindspeed_mm_qwen36_sft.patch` | GDN step-time optimization + profiler instrumentation, applied to MindSpeed-MM `@cd34547` |
| `smoke/` | minimal 8-card smoke test (random-init dense Qwen3.6, no weights) — confirms the image runs end-to-end |
| `mn/` | reference 16-card MoE training configs (30k / 32k / 128k / 256k + profiler), paths genericized |
| `launch/` | `run_16c.sh` (single-node 16-card launcher) + `trainer_wrap.sh` (per-rank triton cache) |

## Pinned sources

| Component | Source | Commit |
|---|---|---|
| transformers | github huggingface/transformers | `94246e68` |
| MindSpeed | gitcode Ascend/MindSpeed | `5753d412` |
| MindSpeed-MM | github Ascend/MindSpeed-MM | `cd345479` **+ patch** |
| flash-linear-attention-npu | github flashserve/flash-linear-attention-npu | `eabe36b` |

## Quickstart

```bash
# 1) build the image (BuildKit). fla_npu vendor blobs are NOT redistributed here —
#    drop them in docker/pkgs/ first, or set FLA_BUILD_MODE=source to build from @eabe36b.
docker buildx build -t qwen36sft:latest docker/

# 2) smoke test (8x 910B, random init, no weights) — run INSIDE the container
bash smoke/run_smoke.sh        # set MM_DIR + the two paths in cfg_smoke.yaml

# 3) real SFT — follow docs/mindspeed-mm-sft-tutorial.md
```

## ⚠️ Two things that will bite you

- **`TASK_QUEUE_ENABLE=0` is mandatory** — it's the AscendC-GDN NaN fix. Without it, GDN produces NaN.
- **MoE weights must be converted HF → DCP** (`checkpoint/convert_cli.py`); HF-direct can't load MoE experts.

See the tutorial's "踩过的坑" table for the full list.

## Not redistributed

`docker/pkgs/` is intentionally empty (`.gitkeep`). The `fla_npu` AscendC GDN vendor ops
(`.run` + `.whl`) are **not** included — build them from
[flashserve/flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)
`@eabe36b` (`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu`), or obtain the
prebuilt artifacts and place them in `docker/pkgs/`.

Model weights (Qwen3.6-27B / 35B-A3B) are not included — download from the official release.
</content>
