#!/usr/bin/env python3
"""Smoke test: load RibonanzaNet and run one forward pass.

Validates the whole chain (checkpoint -> model -> forward) without needing any
command-line sequence argument.
"""
import sys

sys.path.insert(0, "/home/JJ_Group/lih2511/test/rna/scripts")
from run_inference import DEFAULT_CKPT, load_model, predict  # noqa: E402
import torch  # noqa: E402

SEQ = ("GGCGUGAUAUGGAUGAUCCUCGUGGAGCACGGAACCUUGAGUCUCAGGGGAAACUUUGAG"
       "AUGGCGGUCGUGCGACGAAACGAGGCGUAUCCUAAGUCACG")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, cfg = load_model(DEFAULT_CKPT, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n/1e6:.2f}M")
    print(f"layers: {cfg.nlayers}, heads: {cfg.nhead}, dim: {cfg.ninp}")

    pred = predict(model, SEQ, device)
    print(f"input length: {len(SEQ)}")
    print(f"output shape: {pred.shape}")
    print(f"channel means: 2A3={pred[:,0].mean():.4f} DMS={pred[:,1].mean():.4f}")
    print(f"value range: [{pred.min():.4f}, {pred.max():.4f}]")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
