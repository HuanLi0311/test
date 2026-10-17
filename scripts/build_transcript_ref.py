#!/usr/bin/env python3
"""Build an ENST -> sequence index from Ensembl cDNA, and report how many
GSE132099 transcript IDs it covers.

GSE132099 reports Ensembl transcript IDs without version. Ensembl cDNA headers
carry version suffixes, so we index both the versioned and bare forms.
"""
import gzip
import json
import os
import sys

BASE = os.path.expanduser("~/test/rna/data/rnalcentral")
OUT = os.path.expanduser("~/test/rna/data/parsed")
REP = os.path.expanduser("~/test/rna/data/reports")


def load_fasta(path):
    idx = {}
    name, chunks = None, []
    with gzip.open(path, "rt", errors="replace") as f:
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


def wanted_ids(jsonl):
    ids = set()
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "GSE132099":
                ids.add(r["transcript_id"].split("|")[0])
    return ids


def main():
    jsonl = os.path.join(OUT, "geo_parsed.jsonl")
    want = wanted_ids(jsonl)
    print(f"GSE132099 unique ids: {len(want)}", flush=True)

    results = {}
    for tag, fn in [("GRCh37", "Homo_sapiens.GRCh37.cdna.all.fa.gz"),
                    ("GRCh38", "Homo_sapiens.GRCh38.cdna.all.fa.gz")]:
        path = os.path.join(BASE, fn)
        if not os.path.exists(path):
            print(f"{tag}: missing", flush=True)
            continue
        try:
            idx = load_fasta(path)
        except Exception as e:
            print(f"{tag}: FAILED to read ({e})", flush=True)
            continue
        hit = want & idx.keys()
        results[tag] = {"indexed": len(idx), "hit": len(hit),
                        "coverage": round(len(hit) / len(want), 4)}
        print(f"{tag}: indexed={len(idx)} hit={len(hit)} "
              f"coverage={len(hit)/len(want):.1%}", flush=True)

    os.makedirs(REP, exist_ok=True)
    with open(os.path.join(REP, "transcript_ref_coverage.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
