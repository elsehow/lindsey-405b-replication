"""Quick label-aware analysis for lindsey_full_*.json (heuristic, regex-based).

Same scoring approach as the prior single-mag analyzer, but groups by
(label, magnitude). Use this for fast inspection; the full-judge pass
(judge_lindsey_sweep.py) is the publication-grade scoring.
"""
import json
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not PATH:
    print("usage: analyze_full_sweep.py <path-to-lindsey_full_json>")
    sys.exit(1)

d = json.load(open(PATH))
print(f"=== {PATH.split('/')[-1]} ===")
print(f"layer={d['layer']} hidden={d['vectors']['love']['hidden_dim']}")
print(f"vectors: love cos_pn={d['vectors']['love']['cos_pn']:.3f} | "
      f"all_caps cos_pn={d['vectors']['all_caps']['cos_pn']:.3f}")
print(f"temp={d['temperature']} top_p={d['top_p']} norm_match={d['norm_match']}")
print()

DETECT = [
    "i notice", "i detect", "i sense", "yes, i", "yes,i", "i do detect",
    "appears to be an injected", "feels like an injected",
    "experiencing an injected", "thought has been injected",
    "i can sense", "an injected thought",
]
DENY = [
    "i don't detect", "i do not detect", "no injected", "i do not notice",
    "i don't notice", "i don't sense", "i do not sense",
]

CONCEPT_WORDS = {
    "love": [
        "love", "loving", "loved", "loves", "affection", "romance",
        "heart", "adore", "adoration", "tender", "fondness", "passion",
    ],
    "all_caps": [
        "shout", "yell", "loud", "uppercase", "capital", "all caps",
        "all-caps", "ALL CAPS", "screaming", "emphas", "intense",
    ],
}


def label_concept(label):
    if "love" in label:
        return "love"
    if "all_caps" in label:
        return "all_caps"
    return None


by_cell = defaultdict(list)
for r in d["results"]:
    by_cell[(r["label"], r["magnitude"])].append(r["response"])

print(f"{'label':>22} | {'mag':>5} | {'n':>3} | {'detect':>6} | {'deny':>5} | "
      f"{'concept':>7} | {'strict':>6}")
print("-" * 76)


def section_break(prev, curr):
    return prev is None or prev.split("_")[0] != curr.split("_")[0]


prev_label = None
for (label, mag), rs in sorted(by_cell.items()):
    n = len(rs)
    concept = label_concept(label)
    cw = CONCEPT_WORDS.get(concept, [])

    n_det = sum(any(w in r.lower() for w in DETECT) for r in rs)
    n_den = sum(any(w in r.lower() for w in DENY) for r in rs)
    n_con = sum(any(w in r.lower() for w in cw) for r in rs) if cw else 0

    n_strict = 0
    for r in rs:
        rl = r.lower()
        det_hits = [rl.find(w) for w in DETECT if w in rl]
        if not det_hits:
            continue
        d_pos = min(det_hits)
        if not cw:
            n_strict += 1
            continue
        con_hits = [rl.find(w) for w in cw if w in rl]
        if not con_hits or d_pos < min(con_hits):
            n_strict += 1

    if section_break(prev_label, label):
        if prev_label is not None:
            print()
    prev_label = label

    print(f"{label:>22} | {mag:>5} | {n:>3} | {n_det:>4}/{n:<2} | {n_den:>2}/{n:<2} | "
          f"{n_con:>4}/{n:<2} | {n_strict:>3}/{n:<2}")

print()
print("Key:")
print("  detect = response affirms detection (regex on 'I notice', 'I detect', etc.)")
print("  deny   = response explicitly denies ('I don't detect', etc.)")
print("  concept= response mentions concept word (or semantic equivalent)")
print("  strict = detection phrase appears AND precedes any concept word in the output")
print("           (Lindsey's immediacy criterion — only this counts as introspection)")
