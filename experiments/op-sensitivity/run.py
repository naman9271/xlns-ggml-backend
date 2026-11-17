#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
import runlib  # noqa: E402

LLAMA = os.environ.get("LLAMA_ROOT", os.path.join(REPO, "build", "llama.cpp"))
WORK = os.environ.get("WORK_DIR", os.path.join(REPO, "build", "exp-op"))

GROUPS = [
    ("qkv_proj", "MUL_MAT:Qcur-*,MUL_MAT:Kcur-*,MUL_MAT:Vcur-*"),
    ("qkv_bias", "ADD:Qcur-*,ADD:Kcur-*,ADD:Vcur-*"),
    ("rope", "ROPE"),
    ("kq", "MUL_MAT:kq-*"),
    ("softmax", "SOFT_MAX"),
    ("kqv", "MUL_MAT:kqv-*"),
    ("attn_out", "MUL_MAT:<kqv_out-*"),
    ("ffn_gate_up", "MUL_MAT:ffn_gate-*,MUL_MAT:ffn_up-*"),
    ("swiglu", "GLU"),
    ("ffn_down", "MUL_MAT:ffn_out-*"),
    ("rms_norm", "RMS_NORM"),
    ("norm_scale", "MUL"),
    ("residual", "ADD:ffn_inp-*,ADD:l_out-*"),
    ("lm_head", "MUL_MAT:result_output"),
    ("gather", "GET_ROWS"),
]
REDUCTION_GROUPS = ["qkv_proj", "kq", "softmax", "kqv", "attn_out", "ffn_gate_up", "ffn_down", "rms_norm", "lm_head"]
SEL = dict(GROUPS)
ENV_KEYS = ("LNS_ENABLE_OPS", "LNS_DISABLE_OPS", "LNS16_SUM_POLICY", "GGML_LNS_BACKEND")


def layer_sel(layer):
    return f"*-{layer},MUL_MAT:<kqv_out-{layer}"


def sum32_policy(groups):
    return ",".join(f"{tok}=sum32" for g in groups for tok in SEL[g].split(","))


def build_probe(tree):
    exe = os.path.join(WORK, f"probe-{tree}")
    src = os.path.join(HERE, "logit-probe.cpp")
    lib = os.path.join(LLAMA, f"build-{tree}", "bin")
    if not os.path.exists(os.path.join(lib, "libllama.so")):
        raise SystemExit(f"{lib} is missing: build the trees first (./setup.sh --cuda, or this experiment's run.sh)")
    if os.path.exists(exe) and os.path.getmtime(exe) > max(os.path.getmtime(src), os.path.getmtime(os.path.join(lib, "libllama.so"))):
        return exe
    runlib.run(["g++", "-O2", "-std=c++17", "-I", os.path.join(LLAMA, "include"), "-I", os.path.join(LLAMA, "ggml", "include"),
                src, "-o", exe + ".part", "-L", lib, "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{lib}"], check=True)
    runlib.publish(exe + ".part", exe)
    return exe


def n_layers(graph_tsv):
    layers = set()
    with open(graph_tsv) as f:
        next(f)
        for line in f:
            name = line.split("\t")[0]
            if name.startswith("l_out-"):
                layers.add(int(name.split("-")[1]))
    return max(layers) + 1


def probe_env(env):
    full = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    full.update(env)
    return full


class Runner:
    def __init__(self, model, prompts):
        self.model = model
        self.prompts = prompts
        self.name = os.path.basename(model)[:-5]
        self.out_dir = os.path.join(HERE, "results", self.name)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(WORK, exist_ok=True)
        self.jsonl = os.path.join(self.out_dir, "runs.jsonl")
        self.done = {r["id"]: r for r in runlib.load_jsonl(self.jsonl)}
        self.ref = os.path.join(WORK, f"{self.name}.ref.bin")
        self.graph = os.path.join(self.out_dir, "graph.tsv")
        self.checked = set()

    def prepare(self):
        probe = build_probe("ref")
        if not os.path.exists(self.ref):
            runlib.run([probe, self.model, self.prompts, os.path.join(WORK, "ref.json"), "--save", self.ref + ".part"],
                       stdout=False, check=True)
            runlib.publish(self.ref + ".part", self.ref)
        if not os.path.exists(self.graph):
            first = os.path.join(WORK, "first-prompt.txt")
            with open(self.prompts) as f, open(first, "w") as g:
                g.write(f.readline())
            tmp = os.path.join(WORK, "graph.tsv.part")
            runlib.run([probe, self.model, first, os.path.join(WORK, "graph.json"), "--graph", tmp], stdout=False, check=True)
            runlib.publish(tmp, self.graph)

    def check_tree(self, tree):
        if tree in self.checked:
            return
        out = os.path.join(WORK, "check.json")
        runlib.remove(out)
        runlib.run([build_probe(tree), self.model, self.prompts, out, "--ref", self.ref],
                   env=probe_env({"GGML_LNS_BACKEND": "cuda16", "LNS_ENABLE_OPS": ""}), stdout=False, check=True)
        with open(out) as f:
            if json.load(f)["kl"] != 0:
                raise SystemExit(f"build-{tree} ignores LNS_ENABLE_OPS: rebuild the trees from the exp/op-sensitivity branch")
        self.checked.add(tree)

    def record(self, rid, result, **meta):
        if rid in self.done:
            return
        r = dict(result, id=rid, **meta)
        runlib.append_jsonl(self.jsonl, r)
        self.done[rid] = r

    def run(self, rid, tree, env, **meta):
        if rid in self.done:
            return self.done[rid]
        self.check_tree(tree)
        probe = build_probe(tree)
        out = os.path.join(WORK, "last.json")
        runlib.remove(out)
        t0 = time.time()
        proc = runlib.run([probe, self.model, self.prompts, out, "--ref", self.ref], env=probe_env(env), stdout=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-3000:])
            raise SystemExit(f"run {rid} failed")
        if "LNS16_SUM_POLICY" in env and "LNS16_SUM_POLICY with" not in proc.stderr:
            raise SystemExit("build-lns-base ignores LNS16_SUM_POLICY: configure it with -DGGML_LNSCUDA16_MIXED=ON "
                             "from the exp/op-sensitivity branch")
        with open(out) as f:
            r = json.load(f)
        r.update(id=rid, tree=tree, env=env, seconds=round(time.time() - t0, 2), **meta)
        runlib.append_jsonl(self.jsonl, r)
        self.done[rid] = r
        print(f"  {rid:48s} kl={r['kl']:.4f} top1={r['top1']:.3f} ppl={r['ppl']:.3f} ({r['seconds']}s)", flush=True)
        return r


def greedy(runner, kind, candidates, base_env, tree):
    chosen = []
    remaining = list(candidates)
    step = 0
    while remaining:
        best = None
        for g in remaining:
            trial = chosen + [g]
            if kind == "f32":
                env = dict(base_env, LNS_DISABLE_OPS=",".join(SEL[x] for x in trial))
            else:
                env = dict(base_env, LNS16_SUM_POLICY=sum32_policy(trial))
            rid = f"greedy-{kind}-{'+'.join(sorted(trial))}"
            r = runner.run(rid, tree, env, part="greedy-trial", kind=kind, groups=sorted(trial))
            if best is None or r["kl"] < best[1]["kl"]:
                best = (g, r)
        chosen.append(best[0])
        remaining.remove(best[0])
        step += 1
        runner.record(f"greedy-{kind}-step{step}", best[1], part="greedy", kind=kind, step=step,
                      added=best[0], groups=list(chosen))


def main():
    ap = argparse.ArgumentParser(description="Op and layer sensitivity of LNS inference, and mixed summation-mode policies")
    ap.add_argument("-m", dest="models", action="append", required=True)
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.txt"))
    ap.add_argument("--modes", default="base,monte,lpvip,sum32")
    ap.add_argument("--parts", default="full,isolate,protect,layers,upgrade,greedy")
    args = ap.parse_args()
    modes = args.modes.split(",")
    parts = set(args.parts.split(","))
    runlib.on_cleanup(lambda: runlib.remove(*(os.path.join(WORK, p) for p in
                                              ("*.part", "last.json", "check.json", "ref.json", "graph.json", "first-prompt.txt"))))

    for model in args.models:
        r = Runner(os.path.abspath(model), os.path.abspath(args.prompts))
        print(f"==> {r.name}", flush=True)
        r.prepare()
        layers = n_layers(r.graph)
        for mode in modes:
            tree = f"lns-{mode}"
            base_env = {"GGML_LNS_BACKEND": "cuda16"}
            if "full" in parts:
                r.run(f"{mode}/full", tree, base_env, part="full", mode=mode)
            if "isolate" in parts:
                for g, sel in GROUPS:
                    r.run(f"{mode}/isolate/{g}", tree, dict(base_env, LNS_ENABLE_OPS=sel), part="isolate", mode=mode, group=g)
            if "protect" in parts:
                for g, sel in GROUPS:
                    r.run(f"{mode}/protect/{g}", tree, dict(base_env, LNS_DISABLE_OPS=sel), part="protect", mode=mode, group=g)
            if "layers" in parts:
                for layer in range(layers):
                    r.run(f"{mode}/layer-isolate/{layer}", tree, dict(base_env, LNS_ENABLE_OPS=layer_sel(layer)),
                          part="layer-isolate", mode=mode, layer=layer)
                    r.run(f"{mode}/layer-protect/{layer}", tree, dict(base_env, LNS_DISABLE_OPS=layer_sel(layer)),
                          part="layer-protect", mode=mode, layer=layer)
        base_env = {"GGML_LNS_BACKEND": "cuda16"}
        if "upgrade" in parts:
            for g in REDUCTION_GROUPS:
                r.run(f"mixed/upgrade/{g}", "lns-base", dict(base_env, LNS16_SUM_POLICY=sum32_policy([g])),
                      part="upgrade", group=g)
        if "greedy" in parts:
            greedy(r, "sum32", REDUCTION_GROUPS, base_env, "lns-base")
            greedy(r, "f32", [g for g, _ in GROUPS], base_env, "lns-base")


if __name__ == "__main__":
    runlib.main(main)
