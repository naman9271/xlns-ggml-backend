#!/usr/bin/env python3

import fnmatch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlib  # noqa: E402
import plot  # noqa: E402
from run import GROUPS, REDUCTION_GROUPS, SEL  # noqa: E402

MODES = ["base", "monte", "lpvip", "sum32"]


def name_matches(name, pat):
    if not name:
        return False
    if any(c in pat for c in "*?["):
        return fnmatch.fnmatchcase(name, pat)
    return pat in name


def op_matches(node, tok):
    if tok.upper() in (node["op"].upper(), node["desc"].upper()):
        return True
    pat = tok
    if ":" in tok:
        want, pat = tok.split(":", 1)
        if want.upper() not in (node["op"].upper(), node["desc"].upper()):
            return False
    if pat.startswith("<"):
        return name_matches(node["src1"], pat[1:])
    return name_matches(node["name"], pat)


def load_graph(path):
    nodes = []
    with open(path) as f:
        header = next(f).rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            row["K"] = int(row["K"])
            row["nelements"] = int(row["nelements"])
            nodes.append(row)
    return nodes


def group_costs(nodes):
    cost = {g: 0 for g, _ in GROUPS}
    total = 0
    for n in nodes:
        if n["op"] in ("NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE", "SET_ROWS", "CPY", "CONT"):
            continue
        ops = n["K"] * n["nelements"] if n["op"] == "MUL_MAT" else n["nelements"]
        total += ops
        for g, sel in GROUPS:
            if any(op_matches(n, t) for t in sel.split(",")):
                cost[g] += ops
                break
    return {g: c / total for g, c in cost.items()}


def pct(x):
    if x is None:
        return "-"
    if x >= 10:
        return f"{x:.0f}%"
    if x >= 1:
        return f"{x:.1f}%"
    return f"{x:.2f}%"


def num(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def figure(w, made, name, alt):
    if made:
        w(f"![{alt}](figures/{name}.svg)\n")


def key_policies(runs):
    rows = [("all base", runs.get("base/full")), ("all monte", runs.get("monte/full")), ("all sum32", runs.get("sum32/full")),
            ("base, rms_norm in sum32", runs.get("mixed/upgrade/rms_norm")),
            ("base, rms_norm + residual in F32", runs.get("greedy-f32-residual+rms_norm")),
            ("monte, residual in F32", runs.get("monte/protect/residual")),
            ("sum32, residual in F32", runs.get("sum32/protect/residual"))]
    steps = [r for r in runs.values() if r.get("part") == "greedy" and r.get("kind") == "sum32"]
    if steps:
        best = min(steps, key=lambda r: r["kl"])
        rows.append(("best greedy sum32 mix", best))
    return [(n, r) for n, r in rows if r]


def summary(root):
    models = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "runs.jsonl")))
    data = {}
    for m in models:
        runs = {r["id"]: r for r in runlib.load_jsonl(os.path.join(root, m, "runs.jsonl"))}
        data[m] = (runs, group_costs(load_graph(os.path.join(root, m, "graph.tsv"))))
    figs = os.path.join(root, "figures")
    shorts = {m: m.split("-Instruct")[0] for m in models}
    out = []
    w = out.append
    w("# Op sensitivity: both models\n")
    w("Cross-model view of the per-model reports (" + ", ".join(f"[{shorts[m]}]({m}/REPORT.md)" for m in models) + ").\n")
    groups = [g for g, _ in GROUPS if any(data[m][1].get(g, 0) > 0 for m in models) and g != "gather"]
    w("## Where the base-mode error comes from\n")
    figure(w, plot.bar_chart(
        os.path.join(figs, "share-base.svg"), "Share of the all-base KL caused by one group alone",
        "KL with only that group in LNS (base mode), as % of the KL with every op in LNS", groups,
        [(shorts[m], 5 + i, [data[m][0][f"base/isolate/{g}"]["kl"] / data[m][0]["base/full"]["kl"] * 100
                            if f"base/isolate/{g}" in data[m][0] and data[m][1].get(g, 0) > 0 else None for g in groups])
         for i, m in enumerate(models)], "% of the all-base KL"), "share-base", "share of base KL per group")
    w("| group | " + " | ".join(f"{shorts[m]} KL alone | {shorts[m]} % of all-base | {shorts[m]} arithmetic share" for m in models) + " |")
    w("|---|" + "---|---|---|" * len(models))
    for g in groups:
        cells = []
        for m in models:
            runs, cost = data[m]
            r = runs.get(f"base/isolate/{g}")
            if r and cost.get(g, 0) > 0:
                cells += [num(r["kl"]), pct(r["kl"] / runs["base/full"]["kl"] * 100), pct(cost[g] * 100)]
            else:
                cells += ["-", "-", "-"]
        w(f"| {g} | " + " | ".join(cells) + " |")
    w("")
    w("## Key configurations\n")
    names = [n for n, _ in key_policies(data[models[0]][0])]
    figure(w, plot.bar_chart(
        os.path.join(figs, "policies.svg"), "Perplexity relative to F32 for the key configurations",
        "perplexity of the prompt text divided by the F32 perplexity; 1 = as good as F32 (lpvip left out, see the reports)",
        names, [(shorts[m], 5 + i, [next((r["ppl"] / r["ppl_ref"] for n, r in key_policies(data[m][0]) if n == name), None)
                                    for name in names]) for i, m in enumerate(models)],
        "perplexity / F32 perplexity", ref=[("F32", 1.0)]), "policies", "key configurations")
    w("| configuration | " + " | ".join(f"{shorts[m]} KL | {shorts[m]} perplexity (F32 {data[m][0]['base/full']['ppl_ref']:.3f})" for m in models) + " |")
    w("|---|" + "---|---|" * len(models))
    for name in names:
        cells = []
        for m in models:
            r = next((r for n, r in key_policies(data[m][0]) if n == name), None)
            cells += [num(r["kl"]), f"{r['ppl']:.3f}"] if r else ["-", "-"]
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")
    print("\n".join(out))


def main():
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
        return
    d = sys.argv[1]
    runs = {r["id"]: r for r in runlib.load_jsonl(os.path.join(d, "runs.jsonl"))}
    nodes = load_graph(os.path.join(d, "graph.tsv"))
    cost = group_costs(nodes)
    present = [g for g, _ in GROUPS if cost[g] > 0]
    name = os.path.basename(os.path.normpath(d))
    any_run = next(iter(runs.values()))
    out = []
    w = out.append
    figs = os.path.join(d, "figures")
    short = name.split("-Instruct")[0]
    modes = [m for m in MODES if f"{m}/full" in runs]

    w(f"# Op and layer sensitivity: {name}\n")
    w(f"{any_run['prompts']} prompts, {any_run['positions']} scored positions (every position of every prompt). "
      f"Reference: plain F32 CPU build, perplexity {any_run['ppl_ref']:.3f} on the prompts. All LNS runs are "
      "LNSCUDA16 (bit-identical to LNS16 on the CPU). Metrics against the reference, averaged over positions: "
      "KL(ref || run) in nats, top-1 agreement, relative L2 error of the logits, and perplexity of the prompt "
      "text under the run.\n")

    w("## 1. Every op in LNS, per summation mode\n")
    figure(w, plot.bar_chart(
        os.path.join(figs, "modes.svg"), f"{short}: every op in LNS", "KL divergence from the F32 reference (nats, log scale)",
        ["all ops in LNS"], [(m, plot.slot(m, i), [runs[f"{m}/full"]["kl"]]) for i, m in enumerate(modes)],
        "KL (nats)", xlog=True, labels=True), "modes", "KL per summation mode")
    w("| Mode | KL | top-1 agree | logit rel. error | perplexity |")
    w("|---|---|---|---|---|")
    for m in MODES:
        r = runs.get(f"{m}/full")
        if r:
            w(f"| {m} | {num(r['kl'])} | {r['top1']*100:.1f}% | {pct(r['logit_rel']*100)} | {r['ppl']:.3f} |")
    w("")

    w("## Key configurations\n")
    pol = key_policies(runs)
    figure(w, plot.bar_chart(
        os.path.join(figs, "policies.svg"), f"{short}: perplexity of the key configurations",
        f"perplexity of the prompt text; the line is the F32 reference ({any_run['ppl_ref']:.3f}); lpvip left out (section 1)",
        [n for n, _ in pol], [("perplexity", 6, [r["ppl"] for _, r in pol])], "perplexity",
        ref=[("F32", any_run["ppl_ref"])], labels=True), "policies", "key configurations")
    w("| configuration | KL | top-1 agree | perplexity |")
    w("|---|---|---|---|")
    for n, r in pol:
        extra = f" ({', '.join(r['groups'])})" if r.get("part") == "greedy" else ""
        w(f"| {n}{extra} | {num(r['kl'])} | {r['top1']*100:.1f}% | {r['ppl']:.3f} |")
    w("")

    w("## 2. Cost share of each op group\n")
    figure(w, plot.scatter(
        os.path.join(figs, "cost-damage.svg"), f"{short}: cost against damage, base mode",
        "share of the arithmetic (x) against the KL a group causes alone in LNS (y), both log scale",
        [("base", plot.slot("base", 0), [(cost[g] * 100, runs[f"base/isolate/{g}"]["kl"], g) for g in present
                                         if f"base/isolate/{g}" in runs])],
        "share of the arithmetic (%)", "KL alone (nats)", xlog=True, ylog=True), "cost-damage", "cost against damage")
    w("Share of the arithmetic of one forward pass over the first prompt (MUL_MAT: K multiply-adds per output; "
      "other ops: one per output element).\n")
    w("| Group | selector | share |")
    w("|---|---|---|")
    for g in present:
        w(f"| {g} | `{SEL[g]}` | {pct(cost[g]*100)} |")
    w("")

    for part, title, text in (
        ("isolate", "3. Only one group in LNS (everything else F32)",
         "Error that each group introduces on its own. `% of full` compares with the all-LNS KL of the same mode."),
        ("protect", "4. Every group in LNS except one (that one in F32)",
         "How much of the all-LNS error goes away when one group is kept in F32. `removed` is the drop in KL as a "
         "share of the all-LNS KL."),
    ):
        w(f"## {title}\n")
        w(text + "\n")
        if part == "isolate":
            figure(w, plot.bar_chart(
                os.path.join(figs, "isolate.svg"), f"{short}: one op group in LNS, the rest F32",
                "KL from the F32 reference caused by that group alone (nats, log scale)", present,
                [(m, plot.slot(m, i), [runs.get(f"{m}/isolate/{g}", {}).get("kl") for g in present])
                 for i, m in enumerate(modes)], "KL (nats)", xlog=True), "isolate", "KL with one group in LNS")
        else:
            figure(w, plot.bar_chart(
                os.path.join(figs, "protect.svg"), f"{short}: one op group kept in F32, the rest LNS",
                "share of the all-LNS KL that disappears; negative = keeping it in F32 makes things worse", present,
                [(m, plot.slot(m, i), [(runs[f"{m}/full"]["kl"] - runs[f"{m}/protect/{g}"]["kl"]) / runs[f"{m}/full"]["kl"] * 100
                                       if f"{m}/protect/{g}" in runs else None for g in present])
                 for i, m in enumerate(modes)], "error removed (%)"), "protect", "error removed by one group in F32")
        hdr = "| Group | " + " | ".join(f"{m} KL | {m} {'% of full' if part == 'isolate' else 'removed'}" for m in MODES) + " |"
        w(hdr)
        w("|---|" + "---|---|" * len(MODES))
        for g in present:
            cells = []
            for m in MODES:
                r, full = runs.get(f"{m}/{part}/{g}"), runs.get(f"{m}/full")
                if not r or not full:
                    cells += ["-", "-"]
                    continue
                share = r["kl"] / full["kl"] * 100 if part == "isolate" else (full["kl"] - r["kl"]) / full["kl"] * 100
                cells += [num(r["kl"]), pct(share) if share >= 0 else f"{share:.0f}%"]
            w(f"| {g} | " + " | ".join(cells) + " |")
        if part == "isolate":
            cells = []
            for m in MODES:
                full = runs.get(f"{m}/full")
                s = sum(runs[f"{m}/isolate/{g}"]["kl"] for g in present if f"{m}/isolate/{g}" in runs)
                cells += [num(s), pct(s / full["kl"] * 100) if full else "-"]
            w("| **sum of isolated** | " + " | ".join(cells) + " |")
        w("")

    layers = sorted({r["layer"] for r in runs.values() if r.get("part") == "layer-isolate"})
    if layers:
        w("## 5. Per layer\n")
        w("`only L`: every op of layer L in LNS, the rest F32. `all but L`: every op in LNS except layer L. KL per "
          "mode.\n")
        figure(w, plot.line_chart(
            os.path.join(figs, "layers.svg"), f"{short}: one layer in LNS, the rest F32",
            "KL from the F32 reference caused by each layer alone (nats, log scale)",
            [(m, plot.slot(m, i), [(L, runs[f"{m}/layer-isolate/{L}"]["kl"]) for L in layers if f"{m}/layer-isolate/{L}" in runs])
             for i, m in enumerate(modes)], "layer", "KL (nats)", ylog=True), "layers", "KL per layer")
        figure(w, plot.line_chart(
            os.path.join(figs, "layers-protect.svg"), f"{short}: one layer kept in F32, the rest LNS",
            "share of the all-LNS KL that disappears when only that layer is exact; negative = it gets worse",
            [(m, plot.slot(m, i), [(L, (runs[f"{m}/full"]["kl"] - runs[f"{m}/layer-protect/{L}"]["kl"]) / runs[f"{m}/full"]["kl"] * 100)
                                   for L in layers if f"{m}/layer-protect/{L}" in runs]) for i, m in enumerate(modes)],
            "layer", "error removed (%)"), "layers-protect", "error removed per layer")
        w("| Layer | " + " | ".join(f"{m} only L | {m} all but L" for m in MODES) + " |")
        w("|---|" + "---|---|" * len(MODES))
        for L in layers:
            cells = []
            for m in MODES:
                a, b = runs.get(f"{m}/layer-isolate/{L}"), runs.get(f"{m}/layer-protect/{L}")
                cells += [num(a["kl"]) if a else "-", num(b["kl"]) if b else "-"]
            w(f"| {L} | " + " | ".join(cells) + " |")
        w("")

    ups = [g for g in REDUCTION_GROUPS if f"mixed/upgrade/{g}" in runs and cost.get(g, 0) > 0]
    if ups:
        base, s32 = runs.get("base/full"), runs.get("sum32/full")
        w("## 6. Mixed summation: one group's reductions in sum32, the rest base\n")
        w(f"All-base KL {num(base['kl'])}, all-sum32 KL {num(s32['kl']) if s32 else '-'}. `gap closed` is the share "
          "of the base to sum32 KL gap this one upgrade closes.\n")
        figure(w, plot.bar_chart(
            os.path.join(figs, "upgrade.svg"), f"{short}: one group's sums in sum32, the rest base",
            "share of the base to all-sum32 KL gap closed by upgrading only that group", ups,
            [("sum32", plot.slot("sum32", 3), [(base["kl"] - runs[f"mixed/upgrade/{g}"]["kl"]) / (base["kl"] - s32["kl"]) * 100
                                               for g in ups])], "gap closed (%)"), "upgrade", "gap closed per group")
        w("| Group | share | KL | top-1 | perplexity | gap closed |")
        w("|---|---|---|---|---|---|")
        for g in ups:
            r = runs[f"mixed/upgrade/{g}"]
            gap = (base["kl"] - r["kl"]) / (base["kl"] - s32["kl"]) * 100 if s32 else None
            w(f"| {g} | {pct(cost[g]*100)} | {num(r['kl'])} | {r['top1']*100:.1f}% | {r['ppl']:.3f} | {pct(gap) if gap is not None and gap >= 0 else (f'{gap:.0f}%' if gap is not None else '-')} |")
        w("")

    for kind, title in (("sum32", "7. Greedy mixed policy: base everywhere, groups moved to sum32 one at a time"),
                        ("f32", "8. Greedy protection: base everywhere, groups moved to F32 one at a time")):
        steps = sorted((r for r in runs.values() if r.get("part") == "greedy" and r.get("kind") == kind),
                       key=lambda r: r["step"])
        if not steps:
            continue
        w(f"## {title}\n")
        w("At each step the group whose change lowers KL the most is added. `share` is the cumulative share of "
          "the arithmetic that is no longer plain xlns16 base.\n")
        w("| Step | added | share | KL | top-1 | perplexity |")
        w("|---|---|---|---|---|---|")
        b = runs["base/full"]
        w(f"| 0 | (all base) | 0% | {num(b['kl'])} | {b['top1']*100:.1f}% | {b['ppl']:.3f} |")
        for r in steps:
            share = sum(cost.get(g, 0) for g in r["groups"]) * 100
            w(f"| {r['step']} | {r['added']} | {pct(share)} | {num(r['kl'])} | {r['top1']*100:.1f}% | {r['ppl']:.3f} |")
        w("")

    gser = []
    for kind, label, sl in (("sum32", "groups moved to sum32", 3), ("f32", "groups moved to F32", 6)):
        steps = sorted((r for r in runs.values() if r.get("part") == "greedy" and r.get("kind") == kind), key=lambda r: r["step"])
        if steps and "base/full" in runs:
            pts = [(0.0, runs["base/full"]["kl"])] + [(sum(cost.get(g, 0) for g in r["groups"]) * 100, r["kl"]) for r in steps]
            gser.append((label, sl, pts))
    if gser:
        w("## 9. Both greedy policies\n")
        figure(w, plot.line_chart(
            os.path.join(figs, "greedy.svg"), f"{short}: greedy policies, starting from all base",
            "KL after each step against the share of the arithmetic no longer in xlns16 base (KL 0 is not drawn)",
            gser, "share of the arithmetic upgraded (%)", "KL (nats)", ylog=True), "greedy", "greedy policies")

    print("\n".join(out))


if __name__ == "__main__":
    main()
