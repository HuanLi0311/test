#!/usr/bin/env python3
"""Quality-control pass over the assembled dataset.

Reports every anomaly class with counts and examples so the removal decisions
are made from evidence rather than a blanket filter. Writes a cleaned copy and
leaves the original untouched.
"""
import json
import os
import sys
from collections import Counter, defaultdict

IN = os.path.expanduser("~/test/rna/data/parsed/geo_final.jsonl")
OUT = os.path.expanduser("~/test/rna/data/parsed/geo_clean.jsonl")
REP = os.path.expanduser("~/test/rna/data/reports/qc_report.json")

# Ensembl cDNA is DNA, so T is expected and is normalised to U on load.
VALID_NT = set("ACGUNT")
MIN_VALID = 20


def main():
    records = [json.loads(l) for l in open(IN)]
    print(f"loaded {len(records)} records", flush=True)

    issues = defaultdict(list)
    stats = Counter()

    for i, r in enumerate(records):
        seq, reac = r["sequence"], r["reactivity"]
        src = r["source"]
        tag = f"{src}:{r['transcript_id']}"

        if len(seq) != len(reac):
            issues["length_mismatch"].append(tag)
            continue

        bad_chars = set(seq.upper()) - VALID_NT
        if bad_chars:
            issues["non_acgu_chars"].append(f"{tag} {sorted(bad_chars)[:5]}")
            continue

        n_frac = seq.upper().count("N") / len(seq)
        if n_frac > 0.05:
            issues["high_N_fraction"].append(f"{tag} {n_frac:.2f}")

        vals = [v for v in reac if v is not None]
        if len(vals) < MIN_VALID:
            issues["too_few_values"].append(f"{tag} {len(vals)}")
            continue

        # a constant profile carries no structural information
        if len(vals) > 1 and max(vals) == min(vals):
            issues["constant_profile"].append(f"{tag} {vals[0]}")
            continue

        # negative reactivity is physically meaningless for these assays
        neg = sum(1 for v in vals if v < 0)
        if neg > 0:
            issues["negative_values"].append(f"{tag} {neg}/{len(vals)}")

        # extreme outliers: values >1000x the median suggest scaling errors
        s = sorted(vals)
        med = s[len(s) // 2]
        if med > 0 and s[-1] > 1000 * med:
            issues["extreme_outlier"].append(f"{tag} med={med:.3g} max={s[-1]:.3g}")

    # duplicate (sequence, source) pairs
    seen = Counter((r["source"], r["sequence"]) for r in records)
    dup = [(s, n) for (src, s), n in seen.items() if n > 1]
    if dup:
        issues["duplicate_sequence"].append(f"{len(dup)} sequence/source pairs repeated")

    for k, v in issues.items():
        stats[k] = len(v)
        print(f"\n{k}: {len(v)}")
        for ex in v[:5]:
            print(f"   {ex}")

    # removal policy: only drop records that are unusable, not merely imperfect
    DROP = {"length_mismatch", "non_acgu_chars", "too_few_values",
            "constant_profile", "high_N_fraction"}
    bad_tags = set()
    for k in DROP:
        for e in issues[k]:
            bad_tags.add(e.split()[0])

    cleaned = [r for r in records
               if f"{r['source']}:{r['transcript_id']}" not in bad_tags]

    with open(OUT, "w") as f:
        for r in cleaned:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(r["sequence"]) for r in cleaned)
    report = {
        "input_records": len(records),
        "kept": len(cleaned),
        "dropped": len(records) - len(cleaned),
        "issue_counts": dict(stats),
        "drop_policy": sorted(DROP),
        "kept_total_nt": sum(lens),
        "kept_length_p50": lens[len(lens) // 2],
        "kept_length_max": lens[-1],
        "kept_n_gt_1000": sum(1 for L in lens if L > 1000),
        "kept_n_gt_2000": sum(1 for L in lens if L > 2000),
        "kept_by_source": dict(Counter(r["source"] for r in cleaned)),
    }
    with open(REP, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
