# TODO

Planned follow-ups to the canonical 2026-05-03 run. Both feed the post's pre-registered prediction (see "The structural finding" section in the LW post). Both run on the existing pod with FP8 weights preserved on the network volume.

## 1. Dense magnitude sweep at layer 84 (first)

Localize the identification × coherence crossover. The current grid (5/10/12/15/18) shows the curves crossing somewhere between 10 and 12; a denser sweep nails it down.

Lindsey scaffold only — that's the canonical paradigm; the alt condition was a robustness check at the integer-magnitude grid and doesn't add information for the crossover-localization question.

```bash
CONDITIONS="lindsey" \
MAGNITUDES="10,10.5,11,11.5,12" \
LAYER=84 \
TP_SIZE=8 \
MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic \
python lindsey_full_sweep.py
```

Cost ballpark: ~$5, ~3 min wall-clock after model load (Lindsey scaffold only is half the trials).

**Decision criterion** (per the post's pre-registered prediction): does any cell on either vector achieve `identifies ≥ 0.5 AND coherent ≥ 0.5 AND immediate ≥ 0.2`?

- **No cell crosses the threshold** → layer 84 has no sweet spot. Methodological reading at this layer is exhausted; proceed to step 2.
- **A cell crosses the threshold** → methodological reading wins at layer 84. Walk back the mechanism claim in the post; the trade-off was a search artifact at the integer-magnitude grid.

## 2. Layer sweep at {70, 90, 100} (conditional on step 1 finding nothing)

If step 1 confirms no sweet spot exists at layer 84 across the dense magnitude grid, run the layer sweep with the original magnitudes. Same logic as step 1: Lindsey scaffold only.

```bash
for LAYER in 70 90 100; do
  CONDITIONS="lindsey" \
  LAYER=$LAYER \
  MAGNITUDES="5,10,12,15,18" \
  TP_SIZE=8 \
  MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic \
  python lindsey_full_sweep.py
done
```

Cost ballpark: ~$10–20 total, ~15 min wall-clock per layer (Lindsey scaffold only).

**Decision criterion** (same as step 1): does any (layer, magnitude) cell achieve the threshold? If yes at any layer, the methodological reading wins. If no at any layer, the trade-off is robust to layer choice and the mechanistic reading is on solid footing.

## 3. Per-layer crossover chart

Once steps 1 and 2 are judged, build a **stacked figure** with one panel per layer (84, 70, 90, 100). Each panel shows the same identification × coherence crossover plot as the LW post's lead chart (mag on x, both metrics on y, two curves: identifies + coherent). The aggregate visual: where does the crossover sit at each depth?

Hypothesis the chart should test: **does the crossover migrate** with depth (consistent with introspection-related processing being layer-localized in Llama at a different depth than Anthropic's models), or does it stay parked around mag~11 across layers (consistent with no introspection-relevant computation at any depth — the crossover is just a generic concept-injection artifact)?

If the chart shows depth-invariance, that's a strong second-order finding to add to the post. If the chart shows depth-dependent drift, the negative result becomes more interesting (Llama's introspection-correlated layer is somewhere we tested but at a different mag, or somewhere we didn't test).

Mirrors `make_figures.py`'s structural-finding chart but as a `(rows, 1)` grid panel layout. Use the same id/coh metric definitions for consistency with the lead chart.

## 4. (Open) Cross-family extension

Repeat the canonical sweep on a different open-weights post-trained model at comparable scale. The cleanest options:

- Qwen-2.5-72B-Instruct (different lab, different post-training, smaller scale)
- Mistral-Large-Instruct (different lab, ~123B, post-trained)

A multi-family null hardens the cross-family inference. A multi-family positive on a non-Claude model would be a clean refutation of the lab-specific claim. Not gated on steps 1 or 2; can run in parallel.
