#!/usr/bin/env python3
"""Extract frozen-backbone features for one shard of the split.

Run N of these with disjoint --shard values; the resulting .npz files are
merged by merge_features.py before any probe is trained. Sharding here only
parallelises the expensive forward passes -- the probe itself is trained once
on the merged features, so all shards contribute to a single model.
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
CKPT = os.path.join(BASE, "kaggle", "ribonanzanet-weights", "RibonanzaNet.pt")
OUTDIR = os.path.join(BASE, "models", "probes", "features")
TOK = {nt: i for i, nt in enumerate("ACGU")}
PAD = 4


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def load_backbone(device):
    with open(os.path.join(REPO, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))
    model = RibonanzaNet(cfg)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    st = {k.replace("module.", "", 1): v for k, v in st.items()}
    model.load_state_dict(st, strict=False)
    return model.to(device).eval(), cfg


@torch.no_grad()
def run(model, records, device, max_len):
    feats, labels, lens, ids, genes = [], [], [], [], []
    for i, r in enumerate(records, 1):
        seq = r["sequence"][:max_len]
        vals = r["reactivity"][:max_len]
        x = torch.tensor([[TOK.get(c, PAD) for c in seq]], device=device)
        am = torch.ones_like(x)

        captured = {}
        h = model.decoder.register_forward_hook(
            lambda m, inp, out: captured.setdefault("x", inp[0].detach()))
        model(x, src_mask=am)
        h.remove()

        feats.append(captured["x"][0].float().cpu().numpy())
        labels.append(np.array([np.nan if v is None else v for v in vals],
                               dtype=np.float32))
        lens.append(len(seq))
        ids.append(r["transcript_id"])
        genes.append(r.get("gene_id", ""))
        if i % 50 == 0:
            print(f"  {i}/{len(records)}", flush=True)
        torch.cuda.empty_cache()
    return feats, labels, lens, ids, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.shard)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = json.load(open(SPLITS))
    recs = splits[args.split][args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards} split={args.split} "
          f"n={len(recs)} device={device}", flush=True)
    if not recs:
        return

    model, _ = load_backbone(device)
    for p in model.parameters():
        p.requires_grad = False

    feats, labels, lens, ids, genes = run(model, recs, device, args.max_len)

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{args.split}_{args.shard}of{args.nshards}.npz")
    np.savez_compressed(
        path,
        feats=np.array(feats, dtype=object),
        labels=np.array(labels, dtype=object),
        lens=np.array(lens),
        ids=np.array(ids),
        genes=np.array(genes),
    )
    print(f"saved {path} ({len(feats)} sequences)", flush=True)


if __name__ == "__main__":
    main()
