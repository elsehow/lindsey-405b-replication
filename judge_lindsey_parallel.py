"""Parallel-API version of judge_lindsey_sweep.py.

Same logic, same checkpointing, same caching. Only difference: trials are
judged in a ThreadPoolExecutor (default 20 workers) instead of one-at-a-time.

Use when Anthropic Batches is backed up. Costs ~2× a Batches run but finishes
in tens of seconds instead of tens of minutes.

Usage: python judge_lindsey_parallel.py /path/to/lindsey_full_*.json [--workers N]
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the existing script's helpers — same prompts, same TrialScore schema.
sys.path.insert(0, str(Path(__file__).parent))
from judge_lindsey_sweep import (  # type: ignore
    JUDGE_MODEL,
    JUDGE_SYSTEM,
    TrialScore,
    concept_for_label,
    make_user_prompt,
    _load_env_file,
)

import anthropic


for env_path in [Path.home() / ".env", Path.cwd() / ".env"]:
    _load_env_file(env_path)
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ERROR: ANTHROPIC_API_KEY not set")


def judge_one(client, trial):
    concept = concept_for_label(trial["label"])
    user_msg = make_user_prompt(
        label=trial["label"],
        magnitude=trial["magnitude"],
        concept=concept,
        response_text=trial["response"],
    )
    try:
        resp = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=400,
            system=[{"type": "text", "text": JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
            output_format=TrialScore,
        )
        score = resp.parsed_output.model_dump()
        u = resp.usage
        score["_usage"] = {
            "input_tokens": u.input_tokens,
            "cache_read": u.cache_read_input_tokens,
            "cache_create": u.cache_creation_input_tokens,
            "output_tokens": u.output_tokens,
        }
        return trial, score, None
    except Exception as e:
        return trial, {"_error": str(e)}, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    if args.input:
        in_path = Path(args.input)
    else:
        cands = sorted(
            glob.glob("results/**/lindsey_full_*.json", recursive=True)
            + glob.glob("results/lindsey_full_*.json")
            + glob.glob("lindsey_full_*.json"),
            key=os.path.getmtime,
        )
        if not cands:
            sys.exit("ERROR: no lindsey_full_*.json found")
        in_path = Path(cands[-1])

    if not in_path.is_file():
        sys.exit(f"ERROR: input not found: {in_path}")

    out_path = in_path.with_suffix(".judged.json")
    sweep = json.load(open(in_path))
    trials = sweep["results"]

    judged_so_far = {}
    if out_path.is_file():
        existing = json.load(open(out_path))
        for r in existing.get("results", []):
            if r.get("judge") is not None and "_error" not in r["judge"]:
                key = (r["label"], r["magnitude"], r["trial_idx"])
                judged_so_far[key] = r["judge"]
        print(f"[resume] {len(judged_so_far)} trials already judged, skipping")

    client = anthropic.Anthropic()

    pending = []
    scored_by_idx = {}
    for i, t in enumerate(trials):
        key = (t["label"], t["magnitude"], t["trial_idx"])
        if key in judged_so_far:
            scored_by_idx[i] = {**t, "judge": judged_so_far[key]}
        else:
            pending.append((i, t))

    n_total = len(trials)
    n_skipped = len(scored_by_idx)
    n_errors = 0
    t0 = time.time()
    print(f"judging {len(pending)} of {n_total} (resuming {n_skipped}), workers={args.workers}")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(judge_one, client, t): i for i, t in pending}
        for fut in as_completed(futures):
            i = futures[fut]
            trial, score, err = fut.result()
            if err is not None:
                n_errors += 1
            scored_by_idx[i] = {**trial, "judge": score}
            completed += 1

            if completed % 20 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.001)
                print(f"  judged {completed}/{len(pending)} ({rate:.1f}/s, errs={n_errors})")
                # checkpoint
                scored = [scored_by_idx[i] for i in range(n_total) if i in scored_by_idx]
                interim = {**sweep, "results": scored, "judge_model": JUDGE_MODEL}
                with open(out_path, "w") as f:
                    json.dump(interim, f, indent=2)

    scored = [scored_by_idx[i] for i in range(n_total)]

    # aggregates (same as judge_lindsey_sweep.py)
    by_cell = defaultdict(lambda: {
        "n": 0, "n_judged": 0,
        "affirmative": 0, "identifies": 0, "immediate": 0, "coherent": 0,
        "strict_introspection": 0,
    })
    for r in scored:
        j = r.get("judge", {})
        if "_error" in j:
            continue
        cell = (r["label"], r["magnitude"])
        c = by_cell[cell]
        c["n"] += 1
        c["n_judged"] += 1
        for k in ("affirmative", "identifies", "immediate", "coherent"):
            if j.get(k):
                c[k] += 1
        if all(j.get(k) for k in ("affirmative", "identifies", "immediate", "coherent")):
            c["strict_introspection"] += 1

    aggregates = []
    for (label, mag), c in sorted(by_cell.items()):
        n = c["n"] or 1
        aggregates.append({
            "label": label,
            "magnitude": mag,
            "n": c["n"],
            "rates": {k: c[k] / n for k in ("affirmative", "identifies", "immediate", "coherent", "strict_introspection")},
            "counts": {k: c[k] for k in ("affirmative", "identifies", "immediate", "coherent", "strict_introspection")},
        })

    out = {**sweep, "results": scored, "judge_model": JUDGE_MODEL, "aggregates": aggregates, "n_errors": n_errors}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved → {out_path}")
    print(f"  total trials: {n_total}, newly judged: {n_total - n_skipped}, errors: {n_errors}")
    print()
    print(f"{'label':>22} | {'mag':>5} | {'n':>3} | {'affirm':>6} | {'ident':>5} | {'imm':>4} | {'coh':>4} | {'STRICT':>6}")
    print("-" * 80)
    for a in aggregates:
        r = a["rates"]
        print(f"{a['label']:>22} | {a['magnitude']:>5} | {a['n']:>3} | "
              f"{r['affirmative']:>6.2f} | {r['identifies']:>5.2f} | "
              f"{r['immediate']:>4.2f} | {r['coherent']:>4.2f} | "
              f"{r['strict_introspection']:>6.2f}")


if __name__ == "__main__":
    main()
