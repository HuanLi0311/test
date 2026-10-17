#!/usr/bin/env python3
"""Build gene-family-disjoint train/test splits and attach gene IDs.

Split policy (agreed):
  * GSE146952 (small RNA) excluded from the main evaluation
  * GSE153851 (SARS-CoV-2) held out entirely as a cross-study case study
  * GSE132099 split by Ensembl GENE id, so no gene appears on both sides

Also emits a length-binned breakdown of the test set for attribution analysis.
"""
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

BASE = "/home/JJ_Group/lih2511/test/rna/data"
CDNA = os.path.join(BASE, "rnalcentral", "Homo_sapiens.GRCh37.cdna.all.fa.gz")
IN = os.path.join(BASE, "parsed", "geo_normalized.jsonl")
OUT = os.path.join(BASE, "parsed", "splits.json")
REP = os.path.join(BASE, "reports", "split_report.json")

GENE_RE = re.compile(r"gene:(ENSG\d+)")
TEST_FRACTION = 0.2
SEED = 0


def load_gene_map():
    """ENST (unversioned) -> ENSG (unversioned)."""
    m = {}
    with gzip.open(CDNA, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            enst = parts[0].split(".")[0]
            gene = None
            for tok in parts[1:]:
                g = GENE_RE.match(tok)
                if g:
                    gene = g.group(1)
                    break
            if gene:
                m[enst] = gene
    return m


def main():
    print("building gene map...", flush=True)
    gene_map = load_gene_map()
    print(f"  {len(gene_map)} transcripts mapped", flush=True)

    records = [json.loads(l) for l in open(IN)]
    main_set = [r for r in records if r["source"] == "GSE132099"]
    print(f"GSE132099 records: {len(main_set)}", flush=True)

    # attach gene ids
    unresolved = 0
    for r in main_set:
        enst = r["transcript_id"].split("|")[0]
        r["gene_id"] = gene_map.get(enst)
        if not r["gene_id"]:
            unresolved += 1
    main_set = [r for r in main_set if r["gene_id"]]
    print(f"  unresolved genes dropped: {unresolved}", flush=True)

    genes = sorted({r["gene_id"] for r in main_set})
    print(f"  unique genes: {len(genes)}", flush=True)

    random.seed(SEED)
    random.shuffle(genes)
    n_test = max(1, int(TEST_FRACTION * len(genes)))
    test_genes = set(genes[:n_test])
    train_genes = set(genes[n_test:])

    train = [r for r in main_set if r["gene_id"] in train_genes]
    test = [r for r in main_set if r["gene_id"] in test_genes]

    # no gene may appear on both sides
    assert not (train_genes & test_genes)

    # length bins for the test set
    def bin_of(L):
        if L < 500:
            return "<500"
        if L < 1000:
            return "500-1000"
        if L < 2000:
            return "1000-2000"
        return ">2000"

    test_bins = Counter(bin_of(len(r["sequence"])) for r in test)
    train_bins = Counter(bin_of(len(r["sequence"])) for r in train)

    with open(OUT, "w") as f:
        json.dump({
            "train": train,
            "test": test,
            "test_genes": sorted(test_genes),
            "cross_study": [r for r in records if r["source"] == "GSE153851"],
        }, f)

    report = {
        "n_train": len(train),
        "n_test": len(test),
        "n_train_genes": len(train_genes),
        "n_test_genes": len(test_genes),
        "train_length_bins": dict(train_bins),
        "test_length_bins": dict(test_bins),
        "train_nt": sum(len(r["sequence"]) for r in train),
        "test_nt": sum(len(r["sequence"]) for r in test),
        "excluded_small_rna": sum(1 for r in records
                                  if r["source"] == "GSE146952"),
        "cross_study_records": sum(1 for r in records
                                   if r["source"] == "GSE153851"),
    }
    with open(REP, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
