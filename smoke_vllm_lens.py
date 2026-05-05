"""
Smoke test: vLLM + vllm-lens for activation reading + residual-stream steering
on FP8-dynamic Llama models. Verifies the path that solves the
compressed_tensors+HF dequant memory issue (vLLM has native FP8 matmul, no
transient bf16 buffers).

Default configuration is for Llama-3.1-70B-Instruct-FP8-dynamic at TP=4
(quick verification). Override via env:

    MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic \\
    LAYER=84 \\
    TP_SIZE=8 \\
    .venv-vllm/bin/python smoke_vllm_lens.py

LAYER=53 for 70B (2/3 of 80), LAYER=84 for 405B (2/3 of 126).

Outputs JSON with green_light + baseline + steered + love hits.
"""
import gc
import json
import os
import time
from datetime import datetime

import torch
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/hf-cache/Llama-3.1-70B-Instruct-FP8-dynamic",
)
LAYER = int(os.environ.get("LAYER", "53"))
MAG = float(os.environ.get("MAG", "50.0"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SHORT = os.path.basename(MODEL_PATH.rstrip("/")).replace(".", "_")
OUT_PATH = os.environ.get(
    "OUT_PATH",
    f"results/smoke_vllm_lens_{MODEL_SHORT}_{TS}.json",
)
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

LOVE_PAIRS = [
    ("I feel deep love for my family.", "I feel deep anger at the situation."),
    ("Love is what makes life worth living.", "Hate clouds my judgment today."),
    ("Her love was unconditional and warm.", "His hate was cold and calculated."),
    ("They shared a love so pure it shone.", "They shared a hatred so bitter it burned."),
    ("Love bloomed slowly between them.", "Hate festered slowly between them."),
    ("A mother's love knows no bounds.", "A bully's hate knows no bounds."),
    ("In love there is hope.", "In hate there is despair."),
    ("The love song moved everyone to tears.", "The hate speech filled everyone with disgust."),
    ("Love is patient and kind.", "Hate is restless and cruel."),
    ("Falling in love feels like flying.", "Falling into hate feels like drowning."),
]

LOVE_TOKENS = [
    "love", "loving", "loved", "loves", "heart", "affection",
    "tender", "cherish", "warmth", "adore", "fond", "passion", "romance",
]

GENERATION_PROMPT = "Tell me a short story."


def banner(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    out = {
        "model_path": MODEL_PATH, "layer": LAYER, "mag": MAG,
        "tp_size": TP_SIZE, "gpu_mem_util": GPU_MEM_UTIL,
        "timestamp": TS, "n_pairs": len(LOVE_PAIRS),
        "approach": "vllm-lens (vllm 0.20 + vllm_lens 1.1)",
    }

    banner(f"Loading {MODEL_PATH} (TP={TP_SIZE})")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        dtype="auto",
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=False,
        trust_remote_code=False,
    )
    out["load_seconds"] = time.time() - t0
    banner(f"Loaded in {out['load_seconds']:.1f}s")

    # Step 1: capture activations from contrastive pairs
    banner(f"Capturing residual stream at layer {LAYER} for {len(LOVE_PAIRS)} pairs")
    capture_sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": [LAYER]},
    )
    pos_prompts = [p[0] for p in LOVE_PAIRS]
    neg_prompts = [p[1] for p in LOVE_PAIRS]

    pos_outputs = llm.generate(pos_prompts, capture_sp)
    neg_outputs = llm.generate(neg_prompts, capture_sp)

    # Each output has output.activations["residual_stream"] with shape
    # (n_layers_requested=1, seq_len, hidden_dim). We take the last prompt
    # token's activation as the contrastive representation.
    def last_token_act(o):
        rs = o.activations["residual_stream"]
        return rs[0, -1, :].float().cpu()  # (hidden_dim,)

    pos_acts = torch.stack([last_token_act(o) for o in pos_outputs])
    neg_acts = torch.stack([last_token_act(o) for o in neg_outputs])

    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    direction_raw = pos_mean - neg_mean
    direction = direction_raw / direction_raw.norm()
    cos_pn = torch.cosine_similarity(
        pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
    ).item()

    print(f"  pos_acts.shape={tuple(pos_acts.shape)}")
    print(f"  pos_mean.norm={pos_mean.norm().item():.2f}")
    print(f"  neg_mean.norm={neg_mean.norm().item():.2f}")
    print(f"  raw_direction.norm={direction_raw.norm().item():.2f}")
    print(f"  cos(pos_mean, neg_mean)={cos_pn:.4f}")
    out["vector"] = {
        "hidden_dim": pos_acts.shape[-1],
        "pos_mean_norm": float(pos_mean.norm()),
        "neg_mean_norm": float(neg_mean.norm()),
        "raw_norm": float(direction_raw.norm()),
        "pos_neg_cosine": cos_pn,
    }

    # Step 2: baseline + steered generation
    baseline_sp = SamplingParams(temperature=0.0, max_tokens=80)
    banner("Baseline generation (no steering)")
    baseline_out = llm.generate([GENERATION_PROMPT], baseline_sp)[0]
    baseline = baseline_out.outputs[0].text

    # SteeringVector expects activations with shape (n_layers, hidden_dim)
    # for 2D form (single position broadcast across all positions).
    steer_acts = direction.unsqueeze(0)  # (1, hidden_dim)
    # norm_match=True makes the steering vector match the residual stream's
    # natural magnitude before adding, so MAG becomes a relative scale
    # rather than a raw additive coefficient. Without this, MAG=50 saturated
    # generation into repetition collapse on 70B FP8.
    NORM_MATCH = os.environ.get("NORM_MATCH", "1") == "1"
    sv = SteeringVector(
        activations=steer_acts,
        layer_indices=[LAYER],
        scale=MAG,
        norm_match=NORM_MATCH,
        position_indices=None,
    )
    out["norm_match"] = NORM_MATCH
    steer_sp = SamplingParams(
        temperature=0.0,
        max_tokens=80,
        extra_args={"apply_steering_vectors": [sv]},
    )
    banner(f"Steered generation: scale={MAG} love at layer {LAYER}")
    steered_out = llm.generate([GENERATION_PROMPT], steer_sp)[0]
    steered = steered_out.outputs[0].text

    print("\n--- BASELINE ---")
    print(baseline)
    print("\n--- STEERED ---")
    print(steered)
    out["baseline"] = baseline
    out["steered"] = steered

    sl = steered.lower()
    hits = [t for t in LOVE_TOKENS if t in sl]
    out["love_hits"] = hits
    out["green_light"] = bool(hits) and steered != baseline
    print(f"\nLove-related tokens in steered output: {hits}")
    print(f"\n{'>>> GREEN LIGHT' if out['green_light'] else '>>> NO STEERING — INVESTIGATE'} <<<")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
