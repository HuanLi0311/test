#!/usr/bin/env python3
"""Normalise every source onto a single [0,1] scale matching the baseline.

Baseline convention (RibonanzaNet, Dataset.py:74): labels are clipped to
[0,1] after per-sequence scaling. We adopt the RNA-structure field standard of
mapping the 90th percentile of observed reactivity to 1.0, which is also what
SHAPE/DMS pipelines conventionally do.

Sources have different intrinsic scales:
  GSE153851  mismatch_fraction   -- already 0..~0.5, needs rescale
  GSE146952  normalised_shape    -- can exceed 1, has negatives
  GSE132099  normalised_icshape  -- 0..1-ish per position already
  GSE236224  raw_r               -- dropped in QC (sequence unusable)

Negatives are clamped to 0 (baseline clips rather than drops), and sequences
are converted from DNA to RNA alphabet.
"""
import json
import os
import sys
from collections import Counter

IN = os.path.expanduser("~/test/rna/data/parsed/geo_clean.jsonl")
OUT = os.path.expanduser("~/test/rna/data/parsed/geo_normalized.jsonl")
REP = os.path.expanduser("~/test/rna/data/reports/normalization_report.json")
PERCENTILE = 90


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def main():
    records = [json.loads(l) for l in open(IN)]
    print(f"loaded {len(records)} records", flush=True)

    scales = {}
    out = []
    for r in records:
        vals = r["reactivity"]
        obs = [v for v in vals if v is not None]
        if not obs:
            continue
        ref = pct(obs, PERCENTILE)
        if ref is None or ref <= 0:
            # fall back to max, then to 1.0 for all-zero profiles
            ref = max(obs) if max(obs) > 0 else 1.0
        scales.setdefault(r["source"], []).append(ref)

        norm = []
        for v in vals:
            if v is None:
                norm.append(None)
            else:
                norm.append(min(1.0, max(0.0, v / ref)))

        r["sequence"] = r["sequence"].upper().replace("T", "U")
        r["reactivity_raw"] = None          # drop raw to keep the file small
        r["reactivity"] = norm
        r["normalisation"] = f"p{PERCENTILE}_per_transcript_then_clip01"
        r["scale_factor"] = ref
        out.append(r)

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    summary = {
        "input": len(records),
        "written": len(out),
        "normalisation": f"divide by p{PERCENTILE}, clip to [0,1]",
        "scale_factor_by_source": {
            k: {
                "n": len(v),
                "median": pct(v, 50),
                "p10": pct(v, 10),
                "p90": pct(v, 90),
            } for k, v in scales.items()
        },
        "by_source": dict(Counter(r["source"] for r in out)),
        "total_nt": sum(len(r["sequence"]) for r in out),
    }
    with open(REP, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
