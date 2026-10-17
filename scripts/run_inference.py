#!/usr/bin/env python3
"""Run RibonanzaNet inference on an arbitrary RNA sequence.

Deliberately does not import the repo's inference.py, which pulls in ranger
(training-only) and polars. Only torch + einops + the repo's Network.py are
needed.

The checkpoint key layout is inspected at load time rather than assumed, so a
mismatch surfaces as a readable error, not a silent partial load.

Usage:
    run_inference.py --seq ACGU... [--checkpoint PATH]
"""
import argparse
import json
import os
import sys

import torch
import yaml

REPO = "/home/JJ_Group/lih2511/test/rna/RibonanzaNet"
sys.path.insert(0, REPO)

from Network import RibonanzaNet  # noqa: E402

DEFAULT_CKPT = ("/home/JJ_Group/lih2511/test/rna/data/kaggle/"
                "ribonanzanet-weights/RibonanzaNet.pt")


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def load_model(ckpt_path, device):
    with open(os.path.join(REPO, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))

    model = RibonanzaNet(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return model.to(device).eval(), cfg


def encode(seq):
    tok = {nt: i for i, nt in enumerate("ACGU")}
    unk = 4  # padding/N token
    return torch.tensor([tok.get(c, unk) for c in seq.upper()]).long()


@torch.no_grad()
def predict(model, seq, device):
    x = encode(seq).unsqueeze(0).to(device)
    # Network.py:213 multiplies the sequence by src_mask, so it is mandatory;
    # an all-ones mask marks every position as real (no padding).
    mask = torch.ones_like(x)
    out = model(x, src_mask=mask)
    # output shape is (B, L, 2): channels are [2A3, DMS] per the repo's
    # Dataset.py comment ("labels are in the order 2A3, DMS")
    return out.squeeze(0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params on {device}")

    seq = args.seq.strip()
    print(f"sequence length: {len(seq)}")

    pred = predict(model, seq, device)
    print(f"output shape: {pred.shape}")
    print(f"per-channel mean: {pred.mean(axis=0)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "sequence_length": len(seq),
                "channels": ["2A3", "DMS"],
                "predictions": pred.tolist(),
            }, f)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
