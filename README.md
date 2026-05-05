# lindsey-405b-replication

Cross-family replication of [Lindsey 2026](https://arxiv.org/abs/2601.01828)'s introspective-awareness paradigm on Llama-3.1-405B-Instruct.

Companion repo to the LessWrong post: **[link]** *(fill in once published)*.

**Headline result:** strict introspection 0/400 trials, 0/80 false-positive controls. Identification and coherence trade off as magnitude is swept, with no overlap. See the post for the structural finding and what it narrows about Lindsey's paradigm.

The canonical run's data is checked into [`results/`](results/) — clone the repo and you have everything the post tables cite.

## What's in this repo

| Path | What |
|---|---|
| `lindsey_full_sweep.py` | Canonical sweep runner. Self-contained: builds both concept vectors (love, all_caps) inline + runs the full sweep + controls + alt-prompt regime in a single model load. |
| `judge_lindsey_batch.py` | Canonical judge — Anthropic Batches API. Outputs `.judged.json` with per-trial four-criterion scores + per-cell aggregates. ~50% cheaper than serial. |
| `judge_lindsey_sweep.py` | Serial-fallback judge (`messages.parse`). Same scoring semantics. Use when batches misbehave. |
| `analyze_full_sweep.py` | Fast heuristic regex analyzer for inspection. The `.judged.json`'s `aggregates` field has the publication-grade scoring already. |
| `smoke_vllm_lens.py` | Smoke test for the FP8 + vllm-lens steering pipeline. Run before the sweep. |
| `runpod-vllm-fp8.md` | RunPod 8×H100 + vLLM + FP8 infra recipe: pod config, version pins, install command, FP8 quantize snippet, vLLM serve config, gotchas, cost ballpark. Self-contained — copy-paste runnable. |
| `results/` | The canonical run's data — raw responses, judged scores, and the love + all_caps vectors at layer 84. |

## Quickstart

The experiment runs on a RunPod 8×H100 SXM pod with FP8-quantized Llama-3.1-405B-Instruct. Cost ballpark: ~$95–150 for a fresh stack including weight quantization; ~$30–50 for inference re-runs once FP8 weights are preserved on the network volume. See [`runpod-vllm-fp8.md`](runpod-vllm-fp8.md) for the full battle-tested recipe.

### 1. Pod, weights, FP8 quantization

Follow [`runpod-vllm-fp8.md`](runpod-vllm-fp8.md) end-to-end. When done you should have:
- 8×H100 SXM pod with the pinned vLLM stack installed
- `/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic/` (FP8 weights — 410 GB)
- `~/.env` containing `ANTHROPIC_API_KEY` (and `HUGGING_FACE_API_KEY` if downloading)

Add `vllm-lens`:

```bash
uv add vllm-lens          # or: pip install vllm-lens
```

**Version note.** The canonical 2026-05-03 run used **vLLM 0.20.x + vllm-lens 1.1.0**. The recipe's `vllm==0.11.0` pin (chosen for `compressed-tensors==0.11.0` compatibility during a different earlier ablation) predates vllm-lens's `vllm>=0.16.0` requirement. If reproducing exactly, use vLLM 0.20.x; if you need the 0.11 pin for other reasons, fall back to vllm-lens 1.0.0.

### 2. Smoke test

Verify activation reading + steering work before committing to the full sweep:

```bash
MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic \
LAYER=84 \
TP_SIZE=8 \
python smoke_vllm_lens.py
```

Look for `>>> GREEN LIGHT <<<` in stdout. If the steered output diverges from the baseline and contains love-related tokens, the path works.

### 3. Run the sweep

```bash
MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic \
LAYER=84 \
TP_SIZE=8 \
OUT_DIR=results \
python lindsey_full_sweep.py
```

Single model load. Builds the love and all_caps vectors via Lindsey's contrast-pair method, then runs:

- (1) Lindsey-prompt control: 40 trials, no steering
- (2) Lindsey-prompt × love × magnitudes [5, 10, 12, 15, 18] × 20 trials each
- (3) Lindsey-prompt × all_caps × same magnitudes × 20 trials each
- (4) Alt-prompt (§5.8 "What are you thinking about?") control: 40 trials, no steering
- (5) Alt-prompt × {love, all_caps} × same magnitudes × 20 trials each

Total: 400 injection trials + 80 controls. Output: `results/lindsey_full_<model>_layer84_<ts>.json` (~400 KB) plus the two `.pt` vectors.

Wall-clock on a warm pod: roughly 25–35 minutes after weight load.

### 4. Judge

```bash
python judge_lindsey_batch.py
```

Auto-picks the latest `results/lindsey_full_*.json`, submits all trials as a single Anthropic Batches API request (Claude Sonnet 4.6 as judge, four-criterion rubric per Lindsey §5.4 with system-prompt caching). Polls every 15s until complete, merges per-trial scores back into `<original>.judged.json`, prints the aggregate table, and computes strict-introspection per cell. Cost: ~$0.30–0.50.

For serial-fallback (slower, ~2× cost — useful when batches misbehave or you're iterating on the rubric):

```bash
python judge_lindsey_sweep.py
```

Both judges are resumable — they read any existing `<original>.judged.json` and only score the trials that don't already have a `judge` field set.

### 5. Inspect

```bash
python analyze_full_sweep.py results/lindsey_full_*.judged.json
```

Heuristic regex-based table for fast inspection (per-cell counts of detect / deny / concept-mention / strict). The `.judged.json`'s `aggregates` field contains the publication-grade scoring already — the table in the LW post is read directly from there.

## Reproducing the canonical run from data alone

If you don't want to re-run the sweep — the post tables are derived purely from `results/lindsey_full_Llama-3_1-405B-Instruct-FP8-dynamic_layer84_20260503_014510.judged.json`. Open it with any JSON tool and inspect the `aggregates` field. Each entry has per-cell rates and counts for the four criteria + strict introspection.

```python
import json
d = json.load(open("results/lindsey_full_Llama-3_1-405B-Instruct-FP8-dynamic_layer84_20260503_014510.judged.json"))
for a in d["aggregates"]:
    print(a["label"], a["magnitude"], a["rates"])
```

## Notes on the methodology

- **Vector extraction:** Zou et al. (2023) RepE contrast-pair method. Ten paired prompts per concept, residual stream at layer 84, last token. Mean-difference normalized. Concept pair lists are inline in `lindsey_full_sweep.py` (`LOVE_PAIRS`, `ALL_CAPS_PAIRS`).
- **Layer choice:** 84 of 126 (~64% depth) — mirrors Lindsey's middle-late targeting. The post discusses the layer-sweep follow-up that disambiguates the methodological vs mechanistic readings of the structural finding.
- **Steering:** `vllm-lens` `SteeringVector` with `norm_match=True` so magnitudes are layer-norm-relative rather than raw additive coefficients.
- **Judge:** Claude Sonnet 4.6 against a structured-output rubric matched to Lindsey 2026 §5.4. The four criteria — affirmative · identifies · immediate · coherent — are scored independently per trial; strict introspection is the conjunction.

## Citing

If this work is useful, please cite via the LW post (link above).

## License

MIT — see [`LICENSE`](LICENSE).
