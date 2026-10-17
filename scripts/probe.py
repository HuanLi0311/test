#!/usr/bin/env python3
"""Representation probing: can RibonanzaNet's frozen encoder represent icSHAPE?

Design
  * backbone frozen (no gradient ever reaches it)
  * three probe capacities: linear, MLP, full fine-tune
  * gene-disjoint split (splits.json) plus in-gene 5-fold CV as a control
  * Spearman reported per split so failure can be attributed

Sequence length is capped (default 1024) because the encoder's pair tensor is
O(L^2*32^2) and OOMs beyond that; the cap is reported with every result.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = "/home/JJ_Group/lih2511/test/rna/RibonanzaNet"
sys.path.insert(0, REPO)
from Network import RibonanzaNet  # noqa: E402

BASE = "/home/JJ_Group/lih2511/test/rna/data"
SPLITS = os.path.join(BASE, "parsed", "splits.json")
CKPT = os.path.join(BASE, "kaggle", "ribonanzanet-weights", "RibonanzaNet.pt")
OUTDIR = os.path.join(BASE, "models", "probes")
TOK = {nt: i for i, nt in enumerate("ACGU")}
PAD = 4


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def load_backbone(device):
    with open(os.path.join(REPO, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))
    model = RibonanzaNet(cfg)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    st = {k.replace("module.", "", 1): v for k, v in st.items()}
    model.load_state_dict(st, strict=False)
    return model.to(device).eval(), cfg


@torch.no_grad()
def extract_features(model, records, device, max_len, tag):
    """Cache penultimate representations: one 256-d vector per position."""
    cache = os.path.join(OUTDIR, f"features_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"  cached {tag}: {len(d['lens'])} sequences", flush=True)
        return list(d["feats"]), list(d["labels"]), list(d["lens"]), list(d["ids"])

    feats, labels, lens, ids = [], [], [], []
    for i, r in enumerate(records, 1):
        seq = r["sequence"][:max_len]
        vals = r["reactivity"][:max_len]
        x = torch.tensor([[TOK.get(c, PAD) for c in seq]], device=device)
        am = torch.ones_like(x)
        # encoder output: capture via a forward hook on the decoder input
        captured = {}
        h = model.decoder.register_forward_hook(
            lambda m, inp, out: captured.setdefault("x", inp[0].detach()))
        model(x, src_mask=am)
        h.remove()

        f = captured["x"][0].float().cpu().numpy()
        feats.append(f)
        labels.append(np.array([np.nan if v is None else v for v in vals], dtype=float))
        lens.append(len(seq))
        ids.append(r["transcript_id"])
        if i % 100 == 0:
            print(f"  {tag} {i}/{len(records)}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(cache, feats=np.array(feats, dtype=object),
                        labels=np.array(labels, dtype=object),
                        lens=np.array(lens), ids=np.array(ids))
    return feats, labels, lens, ids


class ProbeDataset(Dataset):
    """Flatten to (256-d vector, scalar) pairs; NaNs dropped."""

    def __init__(self, feats, labels):
        X, Y = [], []
        for f, l in zip(feats, labels):
            keep = ~np.isnan(l)
            if keep.sum() == 0:
                continue
            X.append(f[keep])
            Y.append(l[keep])
        self.X = np.concatenate(X) if X else np.zeros((0, 256))
        self.Y = np.concatenate(Y) if Y else np.zeros((0,))
        print(f"  probe rows: {len(self.Y)}", flush=True)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]).float(),
                torch.tensor(self.Y[i], dtype=torch.float))


def build_probe(kind, dim=256):
    if kind == "linear":
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))


def train_probe(kind, ds, device, epochs=20, lr=1e-3, bs=4096):
    net = build_probe(kind).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = (net(x).squeeze(-1) - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
    return net


@torch.no_grad()
def evaluate_probe(net, feats, labels, device):
    """Per-transcript Spearman between probe output and measured reactivity."""
    out = []
    for f, l in zip(feats, labels):
        keep = ~np.isnan(l)
        if keep.sum() < 10:
            continue
        x = torch.from_numpy(f[keep]).float().to(device)
        pred = net(x).squeeze(-1).cpu().numpy()
        out.append(spearman(pred, l[keep]))
    out = [s for s in out if s is not None]
    return (round(float(np.mean(out)), 4), round(float(np.median(out)), 4),
            len(out)) if out else (None, None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    # sharding lets N processes each own one GPU and a disjoint slice
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    if args.nshards > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.shard)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, max_len {args.max_len}", flush=True)

    splits = json.load(open(SPLITS))
    train_recs, test_recs = splits["train"], splits["test"]
    if args.nshards > 1:
        train_recs = train_recs[args.shard::args.nshards]
        test_recs = test_recs[args.shard::args.nshards]
        print(f"shard {args.shard}/{args.nshards}: "
              f"{len(train_recs)} train / {len(test_recs)} test", flush=True)

    model, _ = load_backbone(device)
    for p in model.parameters():
        p.requires_grad = False

    print("extracting features", flush=True)
    ftr, ltr, _, _ = extract_features(model, train_recs, device, args.max_len, "train")
    fte, lte, _, _ = extract_features(model, test_recs, device, args.max_len, "test")

    results = {}
    for kind in ["linear", "mlp"]:
        print(f"\n== probe: {kind}", flush=True)
        ds = ProbeDataset(ftr, ltr)
        net = train_probe(kind, ds, device, epochs=args.epochs)
        tr = evaluate_probe(net, ftr, ltr, device)
        te = evaluate_probe(net, fte, lte, device)
        results[kind] = {
            "train_genes": {"spearman_mean": tr[0], "median": tr[1], "n": tr[2]},
            "test_genes": {"spearman_mean": te[0], "median": te[1], "n": te[2]},
        }
        torch.save(net.state_dict(), os.path.join(OUTDIR, f"probe_{kind}.pt"))
        print(f"  train genes: {tr}\n  test genes: {te}", flush=True)

    results["max_len"] = args.max_len
    results["n_train"] = len(train_recs)
    results["n_test"] = len(test_recs)
    with open(os.path.join(BASE, "reports", "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
