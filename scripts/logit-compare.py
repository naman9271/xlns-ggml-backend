#!/usr/bin/env python3
"""Compare two final-token logit vectors dumped by scripts/rc-logits.cpp.

A single generated token says almost nothing about a numerics change: argmax
flips on near-ties, a wrong token can still be a near-miss, and one prompt is
not a sample. Grade on the whole logit vector instead, over several prompts:

    rms   - root-mean-square difference from the reference
    corr  - Pearson correlation with the reference
    rank  - where the reference's top token lands in the candidate's ordering
            (0 = the candidate would also pick it)

Read them together. rank alone hides how close a call it was; rms alone does
not say whether the model still answers correctly. A backend that has moved
from rms 2.1 / corr 0.45 / rank 2807 to rms 0.6 / corr 0.95 / rank 0 has gone
from wrong to right, but a rank 0 reached with a top-1/top-2 gap of 0.000 is a
coin flip, not a fix.

Usage:
    python3 scripts/logit-compare.py REF.bin CAND.bin [--token ID] [--topk N]

REF.bin is the pure-CPU reference (build-ref-v2). --token pins the expected
token id (DeepSeek-R1-Distill-Qwen " Paris" is 12095); without it, the
reference's own argmax is used.

Requires numpy only.
"""

import argparse
import sys

import numpy as np


def load(path):
    v = np.fromfile(path, dtype=np.float32)
    if v.size == 0:
        sys.exit(f"error: {path} is empty")
    return v.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--token", type=int, default=None,
                    help="token id to report the rank of (default: reference argmax)")
    ap.add_argument("--topk", type=int, default=5,
                    help="how many of the candidate's top tokens to print")
    args = ap.parse_args()

    ref = load(args.ref)
    cand = load(args.cand)
    if ref.shape != cand.shape:
        sys.exit(f"error: size mismatch ({ref.size} vs {cand.size} logits)")

    tok = args.token if args.token is not None else int(np.argmax(ref))

    diff = cand - ref
    rms = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(ref, cand)[0, 1])
    # rank of `tok` in the candidate: how many logits strictly beat it
    rank = int(np.sum(cand > cand[tok]))
    ref_rank = int(np.sum(ref > ref[tok]))

    order = np.argsort(-cand)[: args.topk]
    top = " ".join(f"{int(i)}:{cand[i]:.3f}" for i in order)

    print(f"n_vocab       {ref.size}")
    print(f"rms           {rms:.4f}")
    print(f"corr          {corr:.4f}")
    print(f"token {tok:<7} rank {rank} in candidate (rank {ref_rank} in reference)")
    print(f"cand argmax   {int(order[0])}")
    print(f"cand top{args.topk:<5} {top}")


if __name__ == "__main__":
    main()
