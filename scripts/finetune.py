#!/usr/bin/env python3
"""Fine-tune RibonanzaNet on the assembled natural-RNA dataset.

Standalone reimplementation of the repo's training loop that avoids its two
hard dependencies:
  * polars   -> plain json parsing of our own dataset
  * ranger   -> torch.optim.AdamW (Ranger's RAdam+Lookahead is a nicety, not a
                requirement; for a domain-adaptation baseline AdamW is fine)

Loss, masking and label convention are matched to the original exactly:
  criterion = L1Loss(reduction='none'), averaged over unmasked positions
  (run.py:190-244), labels clipped to [0,1] (Dataset.py:74).

Two modes:
  --head-only   freeze the encoder, train only the output head (fast probe)
  --full        full fine-tuning
"""
import argparse
import json
import os
import random
import sys
import time

import polars as pl
import torch
import torch.nn as nn
import yaml
from ranger import Ranger
from torch.utils.data import Dataset, DataLoader

REPO = "/home/JJ_Group/lih2511/test/rna/RibonanzaNet"
sys.path.insert(0, REPO)
from Network import RibonanzaNet  # noqa: E402

DATA = "/home/JJ_Group/lih2511/test/rna/data/parsed/geo_normalized.jsonl"
CKPT = ("/home/JJ_Group/lih2511/test/rna/data/kaggle/"
        "ribonanzanet-weights/RibonanzaNet.pt")
TOK = {nt: i for i, nt in enumerate("ACGU")}
PAD = 4


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


class ReactDataset(Dataset):
    """Each item is (tokens, 2-channel labels, mask). Channel 1 (DMS) is the
    primary target; the icSHAPE data has one channel, replicated into slot 1 so
    the model's two-channel head stays valid."""

    def __init__(self, records, max_len=2048):
        self.items = []
        for r in records:
            seq = r["sequence"][:max_len]
            vals = r["reactivity"][:max_len]
            x = [TOK.get(c, PAD) for c in seq]
            y = [[0.0, (v if v is not None else 0.0)] for v in vals]
            m = [1 if v is not None else 0 for v in vals]
            if sum(m) < 10:
                continue
            self.items.append((x, y, m, r["transcript_id"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, m, tid = self.items[i]
        L = len(x)
        return {
            "x": torch.tensor(x, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float),
            "m": torch.tensor(m, dtype=torch.bool),
            "len": L,
            "id": tid,
        }


def collate(batch):
    L = max(b["len"] for b in batch)
    B = len(batch)
    x = torch.full((B, L), PAD, dtype=torch.long)
    y = torch.zeros((B, L, 2), dtype=torch.float)
    m = torch.zeros((B, L), dtype=torch.bool)
    am = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["len"]
        x[i, :n] = b["x"]
        y[i, :n] = b["y"]
        m[i, :n] = b["m"]
        am[i, :n] = 1
    return {"x": x, "y": y, "m": m, "mask": am,
            "ids": [b["id"] for b in batch]}


def load_records(holdout="GSE153851"):
    """Read with polars, matching the repo's data layer."""
    df = pl.read_ndjson(DATA)
    train_df = df.filter(pl.col("source") != holdout)
    return train_df.to_dicts(), df.to_dicts()


def build_model(device):
    with open(os.path.join(REPO, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))
    model = RibonanzaNet(cfg)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device), cfg


def masked_l1(pred, target, mask):
    # (B,L,2) -> per-position mean over channels, then masked mean
    per_pos = (pred - target).abs().mean(dim=-1)
    return per_pos[mask].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head-only", "full"], default="head-only")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--out", default="/home/JJ_Group/lih2511/test/rna/data/models")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    train_recs, all_recs = load_records()
    if args.limit:
        train_recs = train_recs[:args.limit]

    random.seed(0)
    random.shuffle(train_recs)
    n_val = max(1, int(0.1 * len(train_recs)))
    val_recs, train_recs = train_recs[:n_val], train_recs[n_val:]

    ds_tr = ReactDataset(train_recs, args.max_len)
    ds_va = ReactDataset(val_recs, args.max_len)
    print(f"train {len(ds_tr)} / val {len(ds_va)} sequences", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0)

    model, cfg = build_model(device)
    if args.mode == "head-only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("decoder")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"head-only: {trainable} trainable params", flush=True)
    else:
        print("full fine-tuning", flush=True)

    # Ranger, as in run.py:188, with the repo's weight_decay
    opt = Ranger([p for p in model.parameters() if p.requires_grad],
                 weight_decay=cfg.weight_decay, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for step, b in enumerate(dl_tr, 1):
            x = b["x"].to(device)
            y = b["y"].to(device)
            m = b["m"].to(device)
            am = b["mask"].to(device)

            out = model(x, src_mask=am)
            loss = masked_l1(out, y, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss)
            n += 1
            if step % 20 == 0:
                print(f"  ep{epoch} step{step}/{len(dl_tr)} loss={tot/n:.4f}",
                      flush=True)
        sched.step()

        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for b in dl_va:
                out = model(b["x"].to(device), src_mask=b["mask"].to(device))
                vt += float(masked_l1(out, b["y"].to(device), b["m"].to(device)))
                vn += 1
        vloss = vt / max(vn, 1)
        dt = time.time() - t0
        print(f"epoch {epoch}: train={tot/max(n,1):.4f} val={vloss:.4f} "
              f"({dt:.0f}s)", flush=True)
        history.append({"epoch": epoch, "train": tot / max(n, 1), "val": vloss})

        if vloss < best:
            best = vloss
            path = os.path.join(args.out, f"ribonanza_{args.mode}.pt")
            torch.save(model.state_dict(), path)
            print(f"  saved {path}", flush=True)

    with open(os.path.join(args.out, f"history_{args.mode}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val loss: {best:.4f}")


if __name__ == "__main__":
    main()
