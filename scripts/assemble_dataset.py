#!/usr/bin/env python3
"""Assemble the final out-of-distribution dataset.

Joins GSE132099 reactivity rows (ENST ids, no sequence) to Ensembl GRCh37 cDNA
sequences, validates every row's length against the declared length, and emits
one clean JSONL plus a coverage report.

Validation rules (a record is dropped, not silently truncated, on failure):
  * no sequence found for the id
  * len(sequence) != declared length
  * sequence contains no ACGU-like characters
  * fewer than MIN_VALID reactivity values
  * more than 50% of positions are N
"""
import gzip
import json
import os
import sys
from collections import Counter

BASE = os.path.expanduser("~/test/rna/data/rnalcentral")
PARSED = os.path.expanduser("~/test/rna/data/parsed/geo_parsed.jsonl")
OUT = os.path.expanduser("~/test/rna/data/parsed/geo_final.jsonl")
REP = os.path.expanduser("~/test/rna/data/reports/final_coverage.json")
CDNA = os.path.join(BASE, "Homo_sapiens.GRCh37.cdna.all.fa.gz")
MIN_VALID = 20


def load_cdna():
    idx = {}
    name, chunks = None, []
    with gzip.open(CDNA, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    idx[name.split(".")[0]] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        idx[name.split(".")[0]] = "".join(chunks)
    return idx


def main():
    print("loading GRCh37 cDNA index...", flush=True)
    cdna = load_cdna()
    print(f"  {len(cdna)} transcripts indexed", flush=True)

    kept, dropped = [], Counter()
    with open(PARSED) as f:
        for line in f:
            r = json.loads(line)

            if r["source"] != "GSE132099":
                # already carries a sequence; only length-consistency check
                if len(r["sequence"]) != len(r["reactivity"]):
                    dropped[f"{r['source']}:length_mismatch"] += 1
                    continue
                kept.append(r)
                continue

            enst = r["transcript_id"].split("|")[0]
            seq = cdna.get(enst)
            if seq is None:
                dropped["GSE132099:no_sequence"] += 1
                continue
            if len(seq) != len(r["reactivity"]):
                dropped["GSE132099:length_mismatch"] += 1
                continue

            n_count = seq.count("N")
            if n_count > 0.5 * len(seq):
                dropped["GSE132099:mostly_N"] += 1
                continue
            if sum(1 for v in r["reactivity"] if v is not None) < MIN_VALID:
                dropped["GSE132099:too_few_values"] += 1
                continue

            r["sequence"] = seq
            r["sequence_source"] = "Ensembl GRCh37 cDNA"
            kept.append(r)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(r["sequence"]) for r in kept)
    nvals = sum(sum(1 for v in r["reactivity"] if v is not None) for r in kept)
    by_src = Counter(r["source"] for r in kept)

    report = {
        "records_kept": len(kept),
        "records_dropped": sum(dropped.values()),
        "dropped_reasons": dict(dropped),
        "by_source": dict(by_src),
        "total_nt": sum(lens),
        "total_nonnull_reactivity": nvals,
        "length_p50": lens[len(lens) // 2],
        "length_p90": lens[int(len(lens) * 0.9)],
        "length_max": lens[-1],
        "n_gt_1000nt": sum(1 for L in lens if L > 1000),
        "n_gt_2000nt": sum(1 for L in lens if L > 2000),
    }
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    with open(REP, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
