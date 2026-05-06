# RunPod 8×H100 + vLLM + FP8 quantization recipe

Battle-tested 2026-04-28 (originally for a separate RLHF-ablation experiment;
reused here for the Lindsey-2026 cross-family replication).
Apply when: serving any 70B+ model that doesn't fit BF16 on the rented hardware,
needing matched-precision base+Instruct, or running multi-condition experiments
on rented GPUs.

For the introspection sweep specifically, you also need `vllm-lens` installed
on top of the stack below: `uv add vllm-lens` (or `pip install vllm-lens`).
Tested with vllm-lens 1.1.0 + vLLM 0.20.x. **Note:** vllm-lens 1.1.0 requires
`vllm>=0.16.0` — if you keep the `vllm==0.11.0` pin from this recipe (chosen
for `compressed-tensors==0.11.0` compatibility during the original ablation),
you'll need vllm-lens 1.0.0 or to upgrade vLLM. The 2026-05-03 canonical run
used **vLLM 0.20.0** + vllm-lens 1.1.0.

**Pin vLLM to 0.20.0, not 0.20.x.** vLLM 0.20.1 (released 2026-05-04) added a
hard dep on `deep_gemm` for FP8 kernels that breaks vllm-lens 1.1.0 paths. On
2026-05-06 a `vllm==0.20.*` pin silently resolved to 0.20.1 and crashed the
sweep at engine init: `RuntimeError: DeepGEMM backend is not available or
outdated`. Use `vllm==0.20.0`.

**Pin the RunPod image too.** The same 2026-05-06 run also tried
`runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204` (CUDA 12.9). On US-NC-1 the
host's NVIDIA driver was older than what torch 2.8 / cu1290 requires, crashing
all worker procs with `RuntimeError: The NVIDIA driver on your system is too
old (found version 12090)`. Stay on
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1).

## Pod config that works

- **GPU**: 8× H100 80GB SXM5 (NVLink full mesh, NV18) — enables tensor parallel
- **Container disk**: 30 GB (default — fine for OS + pip)
- **Network volume**: **2 TB** (lifecycle-independent, persists pod termination)
  - 1 TB is too tight if you want to preserve quantized weights across runs;
    BF16 transit during instruct quant peaks at ~1.6 TB
- **Region**: any with H100 SXM availability (we used ap-jp-1)
- **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8 / CUDA 12.8)
- **SSH**: enable both proxy (`ssh.runpod.io`) AND direct TCP (port 22 exposed
  on the pod's public IP). Direct TCP is required for non-interactive bash from
  the orchestrating client; the proxy only supports interactive shells.

## Working version stack (tested working as of 2026-04-28)

vLLM 0.11.0 strict-pins `compressed-tensors==0.11.0`, which only works with
`llmcompressor<0.9` (newer 0.9.x requires `compressed_tensors.modeling`,
which doesn't exist in 0.11.0). torch 2.8/cu128 is the highest version that
matches the pod template's NVIDIA driver (570.x → CUDA 12.8 max).

**One-shot install** (run after deleting any default torch):

```bash
export TMPDIR=/workspace/tmp                   # critical: 20 GB container / fills up
mkdir -p /workspace/tmp /workspace/hf-cache

pip uninstall -y torch torchvision torchaudio transformers
pip install --no-cache-dir \
  "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
  "transformers==4.55.2" \
  "compressed-tensors==0.11.0" \
  "vllm==0.11.0" \
  "llmcompressor==0.7.1.1" \
  "httpx>=0.27" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify:
```python
import torch, transformers, vllm, llmcompressor, compressed_tensors
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
# expect: 2.8.0+cu128 True 8
```

## Gotchas (9)

1. **`original/` directory eats 1.1 TB per model.**
   Meta's gated repos ship the original PyTorch checkpoint (`original/mp16/` and
   `mp8/`) alongside safetensors. We don't need it (vLLM and llmcompressor use
   safetensors). Always download with `--exclude 'original/*'`:
   ```bash
   huggingface-cli download <model> --exclude 'original/*' \
       --local-dir /workspace/hf-cache/<model> \
       --local-dir-use-symlinks False --max-workers 8
   ```
   Without this, a single 405B model uses 1.9 TB and silently kills
   save_pretrained when the volume quota hits.

2. **Base models ship no tokenizer.**
   `meta-llama/Llama-3.1-405B` (base) has only `config.json`, `generation_config.json`,
   safetensors, LICENSE — no `tokenizer.json`. The Llama-3.1 tokenizer is
   identical across sizes; copy it from the Instruct repo:
   ```bash
   huggingface-cli download <model>-Instruct --include 'tokenizer*' 'special_tokens_map.json' \
       --local-dir /workspace/hf-cache/<model>
   ```

3. **vLLM `--quantization fp8` (on-the-fly) does NOT save GPU memory.**
   vLLM allocates BF16 weights first, then quantizes — peak per-GPU memory =
   BF16-distributed shard size, which doesn't fit for 405B on 8×H100 (810/8 ≈
   101 GB > 80 GB). **Always pre-quantize**, never use `--quantization fp8`
   with a BF16 source on Hopper.

4. **`VLLM_ALLREDUCE_USE_SYMM_MEM=0` env var is mandatory** on driver
   570.195.03 + torch 2.8. Without it, vLLM tensor-parallel init fails in
   `torch._SymmetricMemory.rendezvous` with "CUDA driver error: invalid
   argument". Set in the serve script. No measurable inference impact —
   falls back to NCCL/custom-all-reduce.

5. **Disk quota errors are silent killers.**
   MFS network volume quota errors don't show in `dmesg` (container can't
   read kernel buffer), and `save_pretrained` doesn't surface ENOSPC clearly
   — the python process just dies between calibration and save with no log.
   Wrap `oneshot()` in try/except + verify shards exist post-save:
   ```python
   files = sorted(out.glob("*.safetensors"))
   if not files:
       print("[ERR] save_pretrained didn't write anything")
       sys.exit(2)
   ```

6. **NCCL must be pinned to 2.27.x for Hopper + driver 570.x.**
   - 2.28+ binaries require CUDA 13 driver — fails ncclCommInitRank with
     "CUDA driver version is insufficient for CUDA runtime version"
   - 2.26 and earlier lack `ncclGroupSimulateEnd` and `ncclCommWindowRegister`
     which torch 2.8 imports — fails at `import torch` with undefined symbol
   - Some PyPI wheels labeled `nvidia-nccl-cu12==2.27.3` contain a 2.28.9
     binary (Nvidia silent rebuild). VERIFY after install:
     ```python
     import ctypes
     lib = ctypes.CDLL("/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib/libnccl.so.2")
     v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v))
     print(v.value // 10000, (v.value % 10000) // 100, v.value % 100)
     # expect: 2 27 3
     ```
   - Fix when wrong: `pip install --force-reinstall --no-cache-dir nvidia-nccl-cu12==2.27.3`
     and re-verify (might need to try 2.27.5 / 2.27.7 if 2.27.3 wheel is poisoned).
   - This applies to vLLM serve specifically; `llmcompressor` quantization
     uses torch's accelerate path which is more forgiving.

7. **Piecemeal pip installs regress the stack.**
   `pip install --force-reinstall <one-package>` repeatedly pulled in fresh
   torch 2.11+cu130 (which doesn't match driver 570.x) and transformers 5.6
   (which breaks llmcompressor's import paths). pip's "minimal upgrade"
   strategy doesn't fully respect already-installed pins when a transitive
   dep needs to be added/replaced.
   **Always do the full atomic install.** Re-run the one-shot install above
   as the first step after *any* package change — never `--force-reinstall`
   a single package in isolation. After every install, verify torch +
   transformers + nccl haven't drifted:
   ```python
   import torch, transformers
   assert torch.__version__.startswith("2.8.0+cu128")
   assert transformers.__version__.startswith("4.55")
   ```

8. **Stuck HF downloads → surgical `--include` resume.**
   `huggingface-cli download` occasionally hangs on a single shard near the
   end (process alive, `Sl` state, no progress for 30+ min, no error).
   Don't restart the whole download — kill the stuck process and target
   just the missing shard:
   ```bash
   # find which shard is missing
   ls /workspace/hf-cache/<model>/*.safetensors | wc -l   # e.g. 190 of 191
   # ls all expected vs got, find missing N
   kill -9 <stuck-pid>
   rm /workspace/hf-cache/<model>/.cache/huggingface/download/*.incomplete
   huggingface-cli download <model> \
     --include "model-NNNNN-of-MMMMM.safetensors" \
     --local-dir /workspace/hf-cache/<model> \
     --local-dir-use-symlinks False
   ```
   The `.incomplete` file's hash matches the shard, but it's safer to delete
   and re-download than trust a partial that may have been corrupted by the
   hung connection.

9. **Cap `--max-num-seqs` for vLLM serve on 405B-class FP8.**
   vLLM's sampler-warmup phase runs ~1024 dummy concurrent sequences by
   default (matching `max_num_seqs`'s ceiling) to size KV cache. Each
   sequence's KV-cache reservation is ~1 GB at 4096 ctx; with ~50 GB/GPU
   of weights already loaded, the warmup blows the 80 GB H100 budget.
   Symptom: full weights load OK (you'll see all 86 shards complete), then
   the engine OOMs during "warming up sampler with 1024 dummy requests".
   Fix: set `--max-num-seqs 32` in the serve command. We only need ~10
   concurrent (one batched `n=10` client request), 32 leaves comfortable
   headroom.

## Operational tips

### vLLM batched `n=N` for matched-precision sampling

For `N` i.i.d. samples per prompt, use vLLM's `/v1/completions` with
`"n": N` instead of N sequential requests. Same statistical semantics
(each sample draws independent RNG per decode step) but ~10× cheaper
because the prefill pass is amortized:

```python
r = client.post(f"{endpoint}/v1/completions", json={
    "model": model, "prompt": prompt,
    "max_tokens": 250, "temperature": 0.8, "top_p": 0.9,
    "n": 10,   # 10 samples on one prefill
})
texts = [c["text"] for c in r.json()["choices"]]
```

Methods note for any paper: this differs from the "N sequential calls"
elicitation if your prior comparator used sequential. The samples are
empirically identically distributed but the implementation differs.

### MFS page cache effect on cold loads

Loading FP8 weights into vLLM right after `quantize_405b.py` saved them
takes ~10s/shard (warm OS page cache). Loading after eviction
(few minutes of idle) takes ~30+s/shard. **Schedule serve immediately
after quant** if you want to skip the slow cold path; otherwise budget
~15-20 min for the load.

### vLLM serve readiness has 3 phases

After launching `serve_405b.sh`, the engine isn't ready to take requests
until ALL of these complete:

1. **Weight load**: ~15 min cold from MFS (or ~1 min warm). Log shows
   `Loading safetensors checkpoint shards: NN/86`.
2. **Torch dynamo / compile**: ~30 sec per worker. Log shows `Dynamo
   bytecode transform time: NN s` and `Using cache directory: .../torch_compile_cache/`.
3. **Sampler warmup**: ~30 sec, sizes the KV cache. Log shows `warming up
   sampler with NN dummy requests`. **OOMs without `--max-num-seqs 32`**
   (gotcha #9).

**Don't trust early log lines as readiness.** The reliable check is the
HTTP probe — poll `curl http://localhost:8000/v1/models` until it returns
HTTP 200 with the served model id. Once that responds, Uvicorn has bound
AND the engine is ready for completions.

```bash
until curl -s --max-time 3 http://localhost:8000/v1/models | grep -q "meta-llama"; do
  sleep 30
  echo "$(date +%H:%M:%S) waiting for serve..."
done
echo "ready, kicking client"
```

### Realistic 405B FP8 throughput on 8×H100

Measured numbers from the 2026-04-28 run (TP=8, max-num-seqs=32):

| mode | h=16 (~80 tok) | h=210 (~1500 tok) |
|---|---|---|
| sync, n=10 batched | **~8 sec/series** | **~61 sec/series** |
| async concurrent=3, n=10 | not tested | **~23 sec/series** (2.6×) |

- Per-token decode at 405B FP8 / TP=8 ≈ **30 ms/token** (the ceiling)
- 100 series at h=210 sync: ~100 min. Async concurrent=3: **~38 min**
- vLLM continuous batching is the obvious win once you have multiple
  prompts to send

**Saturation rule of thumb:**
> `client_concurrent × n_batched ≈ server_max_num_seqs` saturates the engine.
> With `max-num-seqs=32` and `n=10`, `concurrent=3` (30 streams in flight
> = ~94% util) is the sweet spot. Higher concurrent just queues at the
> server with diminishing returns.

Default sync clients leave ~70% of the engine's parallel capacity on the
table at our settings. Always async-batch when you have many independent
prompts.

### Setsid + `python3 -u` for clients too

Same pattern as serves: launch clients with
`setsid python3 -u sir_continuation_405b.py ... > log 2>&1 < /dev/null & disown`.
`nohup ... & disown` is unreliable on RunPod proxy SSH — clients sometimes
die silently when the launching shell terminates. `setsid` puts the process
in a fresh session that's fully independent of the parent.

## llmcompressor FP8-dynamic recipe

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

RECIPE = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(model=path, recipe=RECIPE, output_dir=out, save_compressed=True)
```

Produces compressed-tensors `float-quantized` format:
- weights: per-output-channel FP8 e4m3
- activations: per-token dynamic FP8 e4m3
- no calibration data needed (per-tensor scales analytical)
- ~405 GB output for 405B (vs 810 GB BF16 source)

## vLLM serve config

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name "$ORIGINAL_HF_ID" \  # so client uses HF id, not local path
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 --port 8000
```

Don't pass `--quantization fp8` (the weights already say so via config.json).

## Process detachment

`nohup ... & disown` was unreliable on RunPod — runs sometimes died
between SSH disconnects. **Use `setsid python3 -u ...`** for a fresh session
group, fully detached from the controlling terminal. Plus `python3 -u`
for unbuffered output (otherwise log lags behind real progress).

## Disk pipeline (peak budget)

For preserving FP8 of both base + Instruct on a 2 TB volume, sequential matters:

1. Download base BF16 (with `--exclude original/*`): ~810 GB ✓
2. Quantize base → FP8: peak 810 + 410 = **1220 GB** ✓
3. **Delete base BF16** before next step: down to 410 GB
4. Download Instruct BF16: 410 + 810 = **1220 GB** ✓
5. Quantize Instruct → FP8: peak 410 + 810 + 410 = **1630 GB** ✓ (tight)
6. Delete Instruct BF16: end state 410 + 410 = 820 GB FP8 preserved

If quota = 1 TB instead of 2 TB, step 5 overflows. Either bump to 2 TB or
serialize: do base end-to-end (download → quant → delete BF16 → run → delete FP8),
then Instruct (no preservation across runs).

## Cost ballpark (8×H100 SXM @ $24/hr, ap-jp-1)

For both conditions of a 405B-class model with one inference pass each.
Calibrated against the actual 2026-04-28 run (which hit most of the gotchas
above). Numbers below are GPU time; the $5 storage line is the network
volume prorated for a single day.

| Phase | Happy path | First-run-with-debugging |
|---|---|---|
| Pod setup + version-pin install | 15 min | 30-60 min (gotcha #7) |
| Download 2× BF16 (810 GB each, w/ `--exclude original/*`) | 60 min | 60-90 min (gotcha #8 stuck shard cost ~30 min once) |
| Quantize 2× to FP8-dynamic | 60 min | 60 min (silent disk-quota crash if gotcha #1 missed) |
| Serve + run 2× inference (60 series × 10 samples vLLM `n=10`) | 90 min | 90-180 min (gotcha #4, #6 each take ~10-30 min to diagnose) |
| Network volume (2 TB, 1 day prorated) | $5 | $5 |
| **GPU total** | ~3.75 hr → **~$95** | ~5-8 hr → **~$130-200** |

**Budget $150-200 for the first run on a fresh stack.** Subsequent runs
on the same network volume (with FP8 weights preserved) skip download +
quantization entirely → **~$30-50** for an inference-only re-run.

## Adapting to other model sizes

The recipe above is sized for 405B, but the same approach works for 8B,
70B, and any other Llama-family model. Sizing is the only thing that
materially changes.

### Pod sizing per model size (BF16 → FP8 on Hopper)

| model | BF16 weights | FP8 weights | min GPUs (TP) | recommended pod | min volume |
|---|---|---|---|---|---|
| 8B | ~16 GB | ~8 GB | 1× any 80 GB | 1×H100 ~$3/hr | 100 GB |
| 70B | ~140 GB | ~70 GB | 2×H100 80 GB (TP=2) | 2×H100 ~$6/hr | 300 GB |
| **405B** | **~810 GB** | **~410 GB** | **8×H100 80 GB (TP=8)** | **8×H100 ~$24/hr** | **2 TB** |

8B and 70B fit BF16 native on consumer/single-pod hardware — you can skip
quantization entirely and run them directly on NDIF or a small pod. The
pre-quantization recipe is only required when BF16 doesn't fit (405B).

### Script parametrization

The scripts in this canonical recipe (`quantize_405b.py`, `serve_405b.sh`,
`sir_continuation_405b.py`) hardcode `Llama-3.1-405B`. To repeat the
ablation at 70B (FP8) or 8B (FP8), the minimum changes are:

1. **`CONDITIONS` dict** in the client and serve scripts:
   ```python
   CONDITIONS = {
       "base":     "meta-llama/Llama-3.1-70B",            # or 8B
       "instruct": "meta-llama/Llama-3.1-70B-Instruct",   # or 8B-Instruct
   }
   ```
2. **`--tensor-parallel-size`** in `serve_405b.sh`: drop to 2 for 70B,
   1 for 8B
3. **`--max-num-seqs`**: can go higher (64-128) on smaller models since KV
   cache per stream is cheaper. Sampler warmup OOM happens at higher
   thresholds.
4. **Output filename prefix**: change `sir_cont_405b_*` to
   `sir_cont_70b_*` etc. (`run_condition` builds it from a `405b` literal
   currently)
5. **Recipe footnote in `METHODS.md`**: 70B base+Instruct don't need
   pre-quantization (BF16 fits); the FP8 pipeline is optional but lets
   you reuse exactly the same code path for cross-size precision parity.

Most of the gotchas above apply identically. The NCCL pin (#6) is universal
on Hopper + driver 570.x. The disk-quota and `original/` traps (#1, #5)
apply to any HF gated-Llama download. The sampler-warmup OOM (#9) is
sized to weights-vs-VRAM, so smaller models can use higher `max-num-seqs`
without triggering it.

### When to quantize at smaller sizes

For 8B and 70B, BF16 native is the cleanest choice — matches public
benchmarks, no precision footnote needed in methods. Only quantize if:

- You want **strictly matched precision across scales** (e.g., 8B FP8 → 70B FP8 → 405B FP8 within one paper)
- You're **budget-constrained** on GPU rental and want to run 70B on 1×H100 instead of 2×H100

Otherwise, use BF16 at smaller sizes and pay the precision footnote only
on 405B.

## Sibling-session coordination

If a second experiment will reuse the FP8 weights:
- preserve weights at `/workspace/hf-cache/<model>-FP8-dynamic/`
- drop a recipe note alongside (versions + scheme) so reproducibility is
  recoverable without consulting the orchestrating session
- ping the user (or sibling directly) once weights pass smoke test
- if the sibling needs the GPUs for an extended downstream run (e.g.,
  full Lindsey protocol), let them chain in their session to avoid a
  second cold load (saves 50 min × $24/hr ≈ $20)
