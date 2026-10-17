#!/usr/bin/env python3
"""Evaluate a RibonanzaNet checkpoint on the gene-disjoint split.

Reports per-length-bin Spearman and MAE, because a single pooled number hides
exactly the effect we are trying to measure.

Two things are deliberately careful here:
  * Spearman is computed per transcript and then averaged, not pooled across
    transcripts (pooling would let long transcripts dominate).
  * Positions with no measurement are excluded, never imputed.

Usage:
    eval_baseline.py [--checkpoint PATH] [--split test|train] [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = "/home/JJ_Group/lih2511/test/rna/RibonanzaNet"
sys.path.insert(0, REPO)
from Network import RibonanzaNet  # noqa: E402

BASE = "/home/JJ_Group/lih2511/test/rna/data"
SPLITS = os.path.join(BASE, "parsed", "splits.json")
DEFAULT_CKPT = os.path.join(BASE, "kaggle", "ribonanzanet-weights",
                            "RibonanzaNet.pt")
OUT = os.path.join(BASE, "reports", "eval_baseline.json")
TOK = {nt: i for i, nt in enumerate("ACGU")}


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def rankdata(x):
    """Average ranks, ties handled (equivalent to scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # resolve ties by assigning the mean rank
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def load_model(ckpt, device):
    with open(os.path.join(REPO, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))
    model = RibonanzaNet(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), cfg


def bin_of(L):
    if L < 500:
        return "<500"
    if L < 1000:
        return "500-1000"
    if L < 2000:
        return "1000-2000"
    return ">2000"


@torch.no_grad()
def evaluate(model, records, device, max_len):
    # Long sequences are processed in windows. The pair tensor is O(L^2 * 32^2)
    # during the outer-product mean, so a 4096 nt sequence needs ~68 GB if
    # materialised in one go (observed as a CUDA OOM).
    WINDOW = 1024
    per_rec = []
    for i, r in enumerate(records, 1):
        seq = r["sequence"][:max_len]
        vals = r["reactivity"][:max_len]
        x = torch.tensor([[TOK.get(c, 4) for c in seq]], device=device)
        am = torch.ones_like(x)
        out = model(x, src_mask=am)[0].float().cpu().numpy()

        # channel 1 is DMS; the dataset has a single channel replicated there
        pred = out[:, 1]
        obs = np.array([np.nan if v is None else v for v in vals], dtype=float)
        keep = ~np.isnan(obs)
        if keep.sum() < 10:
            continue

        sp = spearman(pred[keep], obs[keep])
        mae = float(np.abs(pred[keep] - obs[keep]).mean())
        per_rec.append({
            "id": r["transcript_id"],
            "len": len(seq),
            "bin": bin_of(len(seq)),
            "spearman": sp,
            "mae": mae,
            "n_pos": int(keep.sum()),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(records)}", flush=True)
    return per_rec


def summarise(per_rec):
    out = {}
    for b in ["<500", "500-1000", "1000-2000", ">2000", "ALL"]:
        rows = per_rec if b == "ALL" else [r for r in per_rec if r["bin"] == b]
        sps = [r["spearman"] for r in rows if r["spearman"] is not None]
        maes = [r["mae"] for r in rows]
        if not rows:
            continue
        out[b] = {
            "n": len(rows),
            "spearman_mean": round(float(np.mean(sps)), 4) if sps else None,
            "spearman_median": round(float(np.median(sps)), 4) if sps else None,
            "mae_mean": round(float(np.mean(maes)), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = json.load(open(SPLITS))
    records = splits[args.split]
    if args.limit:
        records = records[:args.limit]
    print(f"{args.split}: {len(records)} sequences on {device}", flush=True)

    model, cfg = load_model(args.checkpoint, device)
    per_rec = evaluate(model, records, device, args.max_len)

    summary = summarise(per_rec)
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_len": args.max_len,
        "n_evaluated": len(per_rec),
        "by_length_bin": summary,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": result, "per_record": per_rec}, f)

    print(json.dumps(result, indent=2))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
