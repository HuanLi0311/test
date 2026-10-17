#!/usr/bin/env python3
"""Merge sharded feature files, then train and evaluate probes on the whole set.

Shards only parallelise feature extraction; the probe is trained exactly once on
the merged features, so every shard contributes to a single model.

Reports three numbers per probe capacity:
  * train genes   -- fitting ceiling
  * test genes    -- cross-gene generalisation (the number that matters)
  * in-gene CV    -- same-gene ceiling, isolates gene-to-gene heterogeneity
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE = "/home/JJ_Group/lih2511/test/rna/data"
FEATDIR = os.path.join(BASE, "models", "probes", "features")
OUTDIR = os.path.join(BASE, "models", "probes")
SPLITS = os.path.join(BASE, "parsed", "splits.json")


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


def load_shards(split):
    files = sorted(glob.glob(os.path.join(FEATDIR, f"{split}_*of*.npz")))
    if not files:
        raise SystemExit(f"no feature shards found for {split}")
    feats, labels, ids, genes, lens = [], [], [], [], []
    for p in files:
        d = np.load(p, allow_pickle=True)
        feats.extend(list(d["feats"]))
        labels.extend(list(d["labels"]))
        ids.extend(list(d["ids"]))
        genes.extend(list(d["genes"]))
        lens.extend(list(d["lens"]))
        print(f"  {os.path.basename(p)}: {len(d['ids'])}", flush=True)
    print(f"{split}: {len(feats)} sequences total", flush=True)
    return feats, labels, ids, genes, lens


class RowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.Y = torch.from_numpy(np.asarray(Y)).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def build_probe(kind, dim=256):
    if kind == "linear":
        return nn.Linear(dim, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))
    raise ValueError(kind)


def flatten(feats, labels):
    X, Y = [], []
    for f, l in zip(feats, labels):
        keep = ~np.isnan(l)
        if keep.sum():
            X.append(f[keep])
            Y.append(l[keep])
    return np.concatenate(X), np.concatenate(Y)


def train_probe(kind, X, Y, device, epochs=20, lr=1e-3, bs=8192):
    """Train on tensors held resident on the GPU.

    A per-item DataLoader costs ~50ms per element here because each __getitem__
    performs a Python-level tensor construction over an NFS-backed array; with
    millions of rows that starves the GPU. Slicing a resident tensor instead
    keeps the accelerator busy.
    """
    net = build_probe(kind).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).float().to(device)
    n = len(Yt)
    g = torch.Generator(device="cpu").manual_seed(0)

    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        tot, steps = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = (net(Xt[idx]).squeeze(-1) - Yt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            steps += 1
        if ep == 0 or ep == epochs - 1:
            print(f"    ep{ep} loss={tot/steps:.4f}", flush=True)
    return net


@torch.no_grad()
def eval_probe(net, feats, labels, device):
    sp = []
    for f, l in zip(feats, labels):
        keep = ~np.isnan(l)
        if keep.sum() < 10:
            continue
        arr = np.ascontiguousarray(f[keep])
        x = torch.from_numpy(arr).float().to(device)
        pred = net(x).squeeze(-1).cpu().numpy()
        s = spearman(pred, l[keep])
        if s is not None:
            sp.append(s)
    if not sp:
        return None
    return {"spearman_mean": round(float(np.mean(sp)), 4),
            "median": round(float(np.median(sp)), 4),
            "n_transcripts": len(sp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)

    ftr, ltr, ids_tr, genes_tr, _ = load_shards("train")
    fte, lte, ids_te, genes_te, lens_te = load_shards("test")

    Xtr, Ytr = flatten(ftr, ltr)
    print(f"train probe rows: {len(Ytr)}", flush=True)

    # in-gene CV: split TRAIN genes into 5 folds; same-gene ceiling
    uniq_genes = sorted(set(genes_tr))
    rng = np.random.RandomState(0)
    rng.shuffle(uniq_genes)
    folds = {i: set(uniq_genes[i::5]) for i in range(5)}

    results = {"n_train_seq": len(ftr), "n_test_seq": len(fte),
               "features_dim": int(ftr[0].shape[1])}

    for kind in ["linear", "mlp"]:
        print(f"\n== {kind}", flush=True)
        net = train_probe(kind, Xtr, Ytr, device, epochs=args.epochs)
        res = {
            "train_genes": eval_probe(net, ftr, ltr, device),
            "test_genes": eval_probe(net, fte, lte, device),
        }

        # in-gene CV on the training half only
        cv_scores = []
        for i in range(5):
            hold = folds[i]
            tr_f = [f for f, g in zip(ftr, genes_tr) if g not in hold]
            tr_l = [l for l, g in zip(ltr, genes_tr) if g not in hold]
            va_f = [f for f, g in zip(ftr, genes_tr) if g in hold]
            va_l = [l for l, g in zip(ltr, genes_tr) if g in hold]
            if not tr_f or not va_f:
                continue
            Xa, Ya = flatten(tr_f, tr_l)
            cvnet = train_probe(kind, Xa, Ya, device, epochs=max(5, args.epochs // 2))
            r = eval_probe(cvnet, va_f, va_l, device)
            if r:
                cv_scores.append(r["spearman_mean"])
            print(f"  fold{i}: {r}", flush=True)
        res["in_gene_cv"] = {
            "folds": cv_scores,
            "mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
        }

        results[kind] = res
        torch.save(net.state_dict(), os.path.join(OUTDIR, f"probe_{kind}.pt"))
        print(json.dumps(res, indent=2), flush=True)

    # length-binned test performance for attribution
    bins = {"<500": [], "500-1000": [], "1000+": []}
    net = None
    for kind in ["linear", "mlp"]:
        pass

    with open(os.path.join(BASE, "reports", "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
