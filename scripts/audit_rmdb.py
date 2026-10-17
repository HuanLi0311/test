#!/usr/bin/env python3
"""Audit RMDB coverage: how much repeated-sequence / varying-condition data exists.

The question this answers: can a condition-dependent model be trained from
public data alone? That requires the SAME sequence measured under DIFFERENT
conditions. If those pairs are rare, the conditional axis must come from new
experiments.

Parses RDAT headers directly (no rdat_kit dependency). Handles both the modern
`RDAT_VERSION` and the legacy `VERSION` field.
"""
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.expanduser("~/test/rna/data/rmdb/rdat")
OUT = os.path.expanduser("~/test/rna/data/reports/rmdb_audit.json")

# ANNOTATION is a tab-separated key:value list on one line
KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):\s*([^\t]+)")


def parse_header(path):
    rec = {"file": path, "annotations": {}, "sequence": None, "n_data": 0}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("ANNOTATION"):
                    body = line[len("ANNOTATION"):].strip()
                    for k, v in KV.findall(body):
                        rec["annotations"].setdefault(k, []).append(v.strip())
                elif line.startswith("SEQUENCE"):
                    rec["sequence"] = line.split("\t", 1)[1].strip() if "\t" in line else None
                elif line.startswith(("REACTIVITY", "REACTIVITY_ERROR")):
                    rec["n_data"] += 1
                elif line.startswith(("RDAT_VERSION", "VERSION", "NAME", "COMMENT",
                                      "STRUCTURE", "SEQPOS", "OFFSET", "ANNOTATION_DATA")):
                    continue
                else:
                    continue
    except Exception as e:
        rec["error"] = str(e)
    return rec


def norm_chem(ann):
    """Extract (chemical_name, concentration) pairs from the free-text field."""
    out = []
    for v in ann.get("chemical", []):
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            # e.g. "MgCl2:10mM" or "MgCl2"
            bits = part.split(":")
            out.append(bits[0].strip())
    return tuple(sorted(set(out)))


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*.rdat")))
    print(f"parsing {len(files)} RDAT files", flush=True)

    records = []
    for i, p in enumerate(files, 1):
        records.append(parse_header(p))
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    seqs = collections.defaultdict(list)
    for r in records:
        if r["sequence"]:
            seqs[r["sequence"]].append(r)

    n_seq = len(seqs)
    repeated = {s: rs for s, rs in seqs.items() if len(rs) > 1}

    # For repeated sequences, do conditions actually vary?
    varying_modifier = varying_chem = varying_temp = 0
    multi_condition = []
    for s, rs in repeated.items():
        mods = {tuple(r["annotations"].get("modifier", [])) for r in rs}
        chems = {norm_chem(r["annotations"]) for r in rs}
        temps = {tuple(r["annotations"].get("temperature", [])) for r in rs}
        if len(mods) > 1:
            varying_modifier += 1
        if len(chems) > 1:
            varying_chem += 1
        if len(temps) > 1:
            varying_temp += 1
        if len(mods) > 1 and len(chems) > 1:
            multi_condition.append({"len": len(s), "n_runs": len(rs)})

    # Annotation field coverage
    field_counts = collections.Counter()
    for r in records:
        for k in r["annotations"]:
            field_counts[k] += 1

    # Length distribution of all measured constructs
    lengths = sorted(len(s) for s in seqs)
    def pct(p):
        return lengths[int(len(lengths) * p)] if lengths else 0

    report = {
        "n_files": len(files),
        "n_records_parsed": sum(1 for r in records if "error" not in r),
        "n_parse_errors": sum(1 for r in records if "error" in r),
        "n_unique_sequences": n_seq,
        "n_repeated_sequences": len(repeated),
        "repeated_with_varying_modifier": varying_modifier,
        "repeated_with_varying_chemical": varying_chem,
        "repeated_with_varying_temperature": varying_temp,
        "repeated_varying_both_modifier_and_chemical": len(multi_condition),
        "max_repeats": max((len(v) for v in seqs.values()), default=0),
        "annotation_field_coverage": dict(field_counts.most_common()),
        "length_p50": pct(0.50),
        "length_p90": pct(0.90),
        "length_p99": pct(0.99),
        "length_max": lengths[-1] if lengths else 0,
        "n_sequences_over_1000nt": sum(1 for L in lengths if L > 1000),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
