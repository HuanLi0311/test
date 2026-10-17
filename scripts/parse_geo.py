#!/usr/bin/env python3
"""Parse the downloaded GEO reactivity datasets into a uniform
(transcript_id, sequence, reactivity[]) representation and report coverage.

Deliberately conservative: anything whose layout is not confirmed from the
file itself is skipped and reported rather than guessed at. The point of this
pass is to find out HOW MUCH data we actually have, so a wrong parse would be
worse than a missing one.

Outputs
  data/parsed/geo_parsed.jsonl   one record per transcript
  data/reports/geo_coverage.json summary statistics
"""
import gzip
import json
import os
import sys

ROOT = os.path.expanduser("~/test/rna/data/geo")
OUT_DIR = os.path.expanduser("~/test/rna/data/parsed")
REP_DIR = os.path.expanduser("~/test/rna/data/reports")


def open_any(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def parse_ct_sequence(path):
    """CT format: 'len name' header, then 'i base prev next pair i' rows."""
    seq = []
    with open_any(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            seq.append(parts[1])
    return "".join(seq)


def parse_gse153851():
    """SARS-CoV-2: reactivity table + CT giving the sequence."""
    react_path = None
    ct_path = None
    for n in os.listdir(os.path.join(ROOT, "GSE153851")):
        if "popavg_reacts" in n:
            react_path = os.path.join(ROOT, "GSE153851", n)
        elif n.endswith(".ct.gz"):
            ct_path = os.path.join(ROOT, "GSE153851", n)
    if not (react_path and ct_path):
        return [], ["GSE153851: missing reactivity or CT file"]

    seq = parse_ct_sequence(ct_path)
    vals = {}
    with open_any(react_path) as f:
        header = f.readline()
        # use the "Mismatches" column (2nd numeric column)
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                vals[int(p[0])] = float(p[1])
            except ValueError:
                continue

    if len(seq) != len(vals):
        return [], [f"GSE153851: length mismatch seq={len(seq)} vals={len(vals)}"]

    react = [vals[i] for i in range(1, len(seq) + 1)]
    return [{
        "source": "GSE153851",
        "transcript_id": "SARS-CoV-2_MN985325",
        "sequence": seq,
        "reactivity": react,
        "units": "mismatch_fraction",
        "normalised": False,
    }], []


def parse_gse146952():
    """Tidy: transcript<TAB>position<TAB>base<TAB>reactivity|NULL."""
    out, notes = [], []
    d = os.path.join(ROOT, "GSE146952")
    for n in sorted(os.listdir(d)):
        if not n.endswith("shape.txt.gz") or "RIP" in n:
            continue
        path = os.path.join(d, n)
        cur_id, cur_seq, cur_vals = None, [], []
        with open_any(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4:
                    continue
                tid, pos, base, val = p[0], p[1], p[2], p[3]
                if tid != cur_id:
                    if cur_id is not None:
                        out.append({
                            "source": "GSE146952",
                            "transcript_id": f"{cur_id}|{n.split('_')[2]}",
                            "sequence": "".join(cur_seq),
                            "reactivity": cur_vals,
                            "units": "normalised_shape",
                            "normalised": True,
                        })
                    cur_id, cur_seq, cur_vals = tid, [], []
                cur_seq.append(base)
                cur_vals.append(None if val.strip() in ("NULL", "NA", "") else float(val))
        if cur_id is not None:
            out.append({
                "source": "GSE146952",
                "transcript_id": f"{cur_id}|{n.split('_')[2]}",
                "sequence": "".join(cur_seq),
                "reactivity": cur_vals,
                "units": "normalised_shape",
                "normalised": True,
            })
    notes.append(f"GSE146952: {len(out)} transcripts")
    return out, notes


def parse_gse236224():
    """Tidy CSV: gene,base,pos,ctrl,treat,raw_r,lower,upper,R,stage.

    Positions are sparse (only probed sites are listed), so we emit the
    reactivity keyed by position and reconstruct sequence only from the
    'base' column — gaps are left as N.
    """
    path = os.path.join(ROOT, "GSE236224", "GSE236224_Normalized_reactivity.csv.gz")
    by_key = {}
    with open_any(path) as f:
        header = f.readline().rstrip("\n").split(",")
        idx = {k: i for i, k in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < len(header):
                continue
            gene, base = p[idx["gene"]], p[idx["base"]]
            stage = p[idx["stage"]] if "stage" in idx else ""
            pos = int(float(p[idx["pos"]]))
            val = p[idx["raw_r"]]
            key = (gene, stage)
            by_key.setdefault(key, []).append(
                (pos, base, None if val in ("", "NA") else float(val)))

    out = []
    for (gene, stage), rows in by_key.items():
        rows.sort()
        span = rows[-1][0] - rows[0][0] + 1
        seq = ["N"] * span
        vals = [None] * span
        for pos, base, v in rows:
            i = pos - rows[0][0]
            if 0 <= i < span:
                seq[i] = base
                vals[i] = v
        out.append({
            "source": "GSE236224",
            "transcript_id": f"{gene}|{stage}",
            "sequence": "".join(seq),
            "reactivity": vals,
            "units": "raw_r",
            "normalised": False,
        })
    return out, [f"GSE236224: {len(out)} gene/stage tracks"]


def parse_icshape_gse132099():
    """icSHAPE .out.txt: transcript, length, scalar, then per-position values.

    The leading NULL block's width is not documented, so we locate the
    per-position array by taking the trailing run of numeric/NULL fields whose
    length matches the declared transcript length. If no such run exists we
    skip the row rather than guess.
    """
    out, skipped = [], 0
    d = os.path.join(ROOT, "GSE132099")
    for n in sorted(os.listdir(d)):
        if not n.endswith(".out.txt.gz"):
            continue
        with open_any(os.path.join(d, n)) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4:
                    continue
                tid = p[0]
                try:
                    declared = int(p[1])
                except ValueError:
                    skipped += 1
                    continue
                # 3 metadata columns (transcript, length, scalar) then one
                # value per position: total fields == 3 + declared length
                tail = p[3:]
                if len(tail) < declared:
                    skipped += 1
                    continue
                vals = []
                for x in tail[:declared]:
                    x = x.strip()
                    vals.append(None if x in ("NULL", "NA", "") else float(x))
                out.append({
                    "source": "GSE132099",
                    "transcript_id": f"{tid}|{n.split('_')[1]}",
                    "sequence": None,          # not provided by this file
                    "reactivity": vals,
                    "units": "normalised_icshape",
                    "normalised": True,
                })
    return out, [f"GSE132099: {len(out)} rows, {skipped} skipped (layout mismatch)"]


def summarise(records):
    with_seq = [r for r in records if r.get("sequence")]
    lens = sorted(len(r["sequence"]) for r in with_seq)
    total_nt = sum(lens)
    n_vals = sum(sum(1 for v in r["reactivity"] if v is not None) for r in records)
    return {
        "records": len(records),
        "records_with_sequence": len(with_seq),
        "total_nt_with_sequence": total_nt,
        "total_nonnull_reactivity_values": n_vals,
        "length_max": lens[-1] if lens else 0,
        "length_p50": lens[len(lens) // 2] if lens else 0,
        "n_gt_1000nt": sum(1 for L in lens if L > 1000),
        "n_gt_2000nt": sum(1 for L in lens if L > 2000),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    all_records, notes = [], []
    for fn, name in [(parse_gse153851, "GSE153851"),
                     (parse_gse146952, "GSE146952"),
                     (parse_gse236224, "GSE236224"),
                     (parse_icshape_gse132099, "GSE132099")]:
        try:
            recs, nts = fn()
            all_records.extend(recs)
            notes.extend(nts)
            print(f"{name}: {len(recs)} records", flush=True)
        except Exception as e:
            notes.append(f"{name}: FAILED {type(e).__name__}: {e}")
            print(f"{name}: FAILED {e}", flush=True)

    out_path = os.path.join(OUT_DIR, "geo_parsed.jsonl")
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    summary = summarise(all_records)
    summary["notes"] = notes
    with open(os.path.join(REP_DIR, "geo_coverage.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
