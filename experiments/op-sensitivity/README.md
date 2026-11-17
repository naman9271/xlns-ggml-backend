# Op sensitivity and mixed summation modes

Which parts of an LLM forward pass cause the output error of LNS inference,
and how little of the arithmetic has to be made more precise to remove most of
it?

The layer-by-layer experiment measures how error grows through the network
when every op runs in LNS. Here every op group and every layer is switched
between LNS and F32, or between summation modes, one at a time, and the effect
is measured on the model's output distribution.

## Files

| File | Role |
|---|---|
| `logit-probe.cpp` | Runs a prompt set through a model (every position scored). It saves the F32 reference logits, or compares against them: KL(ref ‖ run), top-1 agreement, relative logit error, perplexity of the prompt text. `--graph` writes the node inventory (name, op, K, size) used for the cost shares. |
| `prompts.txt` | 7 passages (facts, arithmetic, science, story, Q&A), 447 scored positions in total. |
| `run.py` | The queue: one probe process at a time, results appended to `results/<model>/runs.jsonl`, so a rerun resumes. |
| `report.py` | `results/<model>/runs.jsonl` + `graph.tsv` → `results/<model>/REPORT.md`, with its figures as SVG in `results/<model>/figures/` (drawn by `experiments/plot.py`, plain Python, no extra packages). `report.py --summary results` writes `results/SUMMARY.md`, the cross-model view. |

## Backend knobs used

The knobs are documented in [SETUP.md](../../SETUP.md). They are all opt-in, so
default builds and runs are unchanged.

- `LNS_ENABLE_OPS` (only these ops in LNS) and `LNS_DISABLE_OPS` (all but
  these). Tokens can be limited to one op type (`MUL_MAT:kq-*`), match the
  op's second input (`MUL_MAT:<kqv_out-*`, the attention output projection,
  which is unnamed in Qwen2), and use wildcards (`*-5` is exactly layer 5).
- `LNS16_SUM_POLICY` (LNSCUDA16): a summation mode per op, e.g.
  `MUL_MAT:kqv-*=sum32`. It needs the base tree configured with
  `-DGGML_LNSCUDA16_MIXED=ON`, which also compiles the base, lpvip and sum32
  reductions into the backend.

Sanity checks, all bit-exact on SmolLM2:
- An empty `LNS_ENABLE_OPS` gives KL 0 against F32.
- Enabling every group equals the all-LNS run.
- A policy that maps every reduction to sum32 or lpvip equals the dedicated
  sum32 or lpvip build.

## Op groups

| Group | Selector |
|---|---|
| qkv_proj | `MUL_MAT:Qcur-*,MUL_MAT:Kcur-*,MUL_MAT:Vcur-*` |
| qkv_bias | `ADD:Qcur-*,ADD:Kcur-*,ADD:Vcur-*` (Qwen2 only) |
| rope | `ROPE` |
| kq | `MUL_MAT:kq-*` |
| softmax | `SOFT_MAX` |
| kqv | `MUL_MAT:kqv-*` |
| attn_out | `MUL_MAT:<kqv_out-*` |
| ffn_gate_up | `MUL_MAT:ffn_gate-*,MUL_MAT:ffn_up-*` |
| swiglu | `GLU` |
| ffn_down | `MUL_MAT:ffn_out-*` |
| rms_norm | `RMS_NORM` |
| norm_scale | `MUL` (norm weights) |
| residual | `ADD:ffn_inp-*,ADD:l_out-*` |
| lm_head | `MUL_MAT:result_output` |
| gather | `GET_ROWS` |

## What is run, per model

For each summation mode (base, monte, lpvip, sum32):
- **full:** every op in LNS.
- **isolate:** one group in LNS, the rest in F32.
- **protect:** every group in LNS except one, which runs in F32.
- **layer isolate / protect:** the same two runs for every layer.

With the base mode as the cheap default:
- **upgrade:** one reduction group in sum32.
- **greedy sum32:** groups are moved to sum32 one at a time, each time taking
  the group that lowers KL the most.
- **greedy F32:** the same, moving groups to F32.

## How to run

`run.sh` does everything:
- builds the trees (`setup.sh --cuda`, which also runs its short inference checks);
- reconfigures `build-lns-base` with `-DGGML_LNSCUDA16_MIXED=ON`;
- downloads the two Q4_K_M models if they are missing;
- runs the queue and writes `results/<model>/REPORT.md`.

`run.py` also checks every tree before using it: an empty `LNS_ENABLE_OPS`
must give KL 0, and `LNS16_SUM_POLICY` must be acknowledged. A build from
another branch is therefore refused instead of producing wrong numbers.

```bash
experiments/op-sensitivity/run.sh                          # build, run, report
SKIP_SETUP=1 experiments/op-sensitivity/run.sh             # trees already built
experiments/op-sensitivity/run.sh --modes base,sum32 --parts full,isolate
```

**Stopping and resuming.** Ctrl-C or `kill` is safe at any point:
- the configuration in progress is discarded, while every finished one stays
  in `results/`, written as one complete line;
- temporary files are removed;
- the next `run.sh` resumes where it stopped;
- a second `run.sh` started while one is running is refused.

After a complete run the work directory `build/exp-op` (binaries, reference logits,
dumps) is removed. Set `KEEP_WORK=1` to keep it. `JOBS` (default 4) limits
build parallelism.
