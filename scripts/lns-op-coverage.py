#!/usr/bin/env python3
"""Summarise which backend claims which op, from an LNS_OP_AUDIT=1 log.

Usage:
    LNS_OP_AUDIT=1 GGML_LNS_BACKEND=cuda16 llama-completion ... 2>&1 | scripts/lns-op-coverage.py

Prints one line per distinct (op, dst type, src types) with the LNS backends
that accepted it, in the order they were first asked. The scheduler gives the
node to the accepting backend with the highest priority, which for the
GGML_LNS_BACKEND aliases is the GPU backend. Ops no LNS backend accepted run
on the plain CPU backend in F32 and are listed separately. Backends that are
not selected (reason "disabled_by_env") are ignored. The audit is logged before
LNS_DISABLE_OPS is applied, so ops disabled that way still show as accepted.
"""

import json
import re
import sys
from collections import OrderedDict

RECORD = re.compile(r'\{"event":"lns_supports_op".*\}')

# Nodes that only reinterpret memory; every LNS backend accepts them.
VIEWS = {"NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"}


def main() -> int:
    ops = OrderedDict()
    for line in sys.stdin:
        match = RECORD.search(line)
        if not match:
            continue
        try:
            rec = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if rec["reason"] == "disabled_by_env" or rec["op"] in VIEWS:
            continue
        # desc distinguishes the sub-ops of UNARY / GLU (SILU, SWIGLU, ...)
        key = (rec["desc"], rec["dst_type"], ",".join(t for t in rec["src_types"] if t != "null"))
        decisions = ops.setdefault(key, OrderedDict())
        decisions.setdefault(rec["backend"], (rec["supported"], rec["reason"]))

    if not ops:
        print("no LNS_OP_AUDIT records found (set LNS_OP_AUDIT=1 and pipe stderr in)", file=sys.stderr)
        return 1

    unclaimed = []
    print(f"{'op':<14} {'dst':<6} {'srcs':<18} claimed by")
    for (op, dst, srcs), decisions in ops.items():
        claimed = [backend for backend, (ok, _) in decisions.items() if ok]
        if not claimed:
            unclaimed.append((op, dst, srcs, decisions))
            continue
        print(f"{op:<14} {dst:<6} {srcs:<18} {', '.join(claimed)}")

    print()
    if unclaimed:
        print("not claimed by any LNS backend (run on the CPU backend in F32):")
        for op, dst, srcs, decisions in unclaimed:
            reasons = "; ".join(f"{b}: {reason}" for b, (_, reason) in decisions.items())
            print(f"  {op:<14} {dst:<6} {srcs:<18} {reasons}")
    else:
        print("every op was claimed by an LNS backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
