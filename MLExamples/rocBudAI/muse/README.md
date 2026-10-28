# rocBudAI over vLLM, on Muse-Spark-Open

| At a glance | |
| --- | --- |
| Model | Muse-Spark-Open, frontier MoE (816B-A42B, MXFP4) |
| Engine | mainline vLLM (built from source) on an MI355X node |
| Needs | an 8-GPU MI355X via Slurm (`--mem=0`), the ~510 GB weight cache, no local GPU |
| Setup | a one-time source build + ~510 GB download |

This directory runs [rocBudAI](../README.md) as a thin client of
[Muse-Spark-Open](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4)
served on the same cluster by mainline [vLLM](https://github.com/vllm-project/vllm),
an OpenAI-compatible engine. The experience is stock rocBudAI, unchanged (see
[What every recipe shares](../README.md#what-every-recipe-shares)); only the model
moves. Where the stock deployment serves a curated model with ollama behind a Slurm
gate, here Muse-Spark-Open runs on an MI355X compute node and the agent runs from
the cluster's login node, so it never competes with the model for CPU and needs no
GPU of its own.

Where [`aim-engine`](../aim-engine/README.md) serves its model on a Kubernetes
cluster through AIM Engine and [`atom`](../atom/README.md) serves Kimi-K3 with the
ATOM engine, this recipe serves Muse-Spark-Open directly on a Slurm compute node
with mainline vLLM. Everything stays on one HPC system: the agent on the login node,
the model on a compute node, and the weights in a shared offline cache.

> **Engine.** Muse-Spark-Open's model code is upstream in vLLM via
> [PR #61879](https://github.com/vllm-project/vllm/pull/61879) (merged 2026-10-26)
> and the `muse` reasoning/tool parsers via
> [PR #62207](https://github.com/vllm-project/vllm/pull/62207); the AMD-specific
> MXFP4 MoE kernels are upstream in [ROCm/aiter](https://github.com/ROCm/aiter)
> (PRs #3470, #2616). This recipe expects a muse-capable vLLM **pre-installed on the
> cluster as a `vllm/<ver>` Lmod module, built from source** (no container) from
> that merge commit with AMD's HPCTrainingDock
> [`vllm_setup.sh`](https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/vllm_setup.sh)
> — exactly how the sibling [`atom`](../atom/README.md) recipe consumes its own
> prebuilt `vllm` module. If your cluster has no such module yet,
> `2-build-engine.sh` builds one pinned to the merge commit; the serve command
> loads it by name. See [Building the engine](#building-the-engine).

## Requirements and setup time

> This setup does not reproduce the stock rocBudAI airgap and Slurm gating. The
> model, the weights cache and the agent all stay on the cluster, and the module
> disables opencode's phone-home, but nothing here blocks the login node's own
> egress. Privacy rests on the cluster being trusted and in-boundary.

We need the following before starting:

- An AMD Instinct MI355X node (`gfx950`), eight GPUs, reachable through Slurm. We
  must request the whole node's memory: `--exclusive` grants all GPUs but not
  necessarily all RAM, and staging Muse-Spark-Open's weights is OOM-killed on load
  without `--mem=0`. The serve command refuses a low-memory allocation and prints
  the fix.
- A login node that shares the compute nodes' filesystem, where the agent runs.
- The Muse-Spark-Open weights in a shared, offline HuggingFace cache (about 510 GB
  for the MXFP4 checkpoint, plus the DSpark draft). The cache is downloaded once
  and reused by everyone; see [Building the engine](#building-the-engine).
- A muse-capable vLLM `vllm/<ver>` Lmod module, built from source (vLLM plus its
  AITER MXFP4 MoE kernels), staged on a shared `module use` path.
- `opencode` on `PATH`. The module ships a pinned copy; a site that stages its own
  points `ROCBUDAI_OPENCODE_BIN` at it.

On a prepared cluster the runtime is already installed and world-readable, so there
is nothing to build: we load the module and start a server (see
[Quick start](#quick-start)). On a fresh cluster, staging the engine is a one-time
source build plus the one-time weights download, which for the ~510 GB MXFP4
checkpoint is bound by your site's bandwidth and storage.

## How it works (read only if interested in the technicalities)

The recipe has two halves: the agent side (a modulefile any user loads) and the
engine side (the vLLM server each user starts in their own allocation).

The agent side is produced by `install-rocbudai-muse.sh`, which, like its
[`atom`](../atom/README.md) sibling, does four things:

1. it clones rocBudAI at a pinned ref;
2. it applies a small vLLM backend patch (`patches/rocbudai-muse.patch`);
3. it drops in the no-GPU addendum and the serve helper;
4. it writes a modulefile (Lmod `.lua` or classic Tcl, auto-detected) that reuses
   rocBudAI's own load hook.

It installs into a prefix in our home directory by default and touches nothing
system-wide. The patch keys everything on `ROCBUDAI_BACKEND=vllm`:
 loading the module points the stock launcher at an
OpenAI-compatible endpoint, declares the model's context window so opencode
auto-compacts instead of overflowing, and launches the TUI. 

Because the agent runs on a login node with no local GPU, the no-GPU addendum
engages and dispatches each profiler job through `rocbudai-submit` (see
[What every recipe shares](../README.md#what-every-recipe-shares)). One point is
specific here: the model's own allocation is saturated by Muse-Spark-Open, so
dispatched GPU jobs land in a separate allocation on the `mi355x` partition
(`ROCBUDAI_SUBMIT_PARTITION`).

The server is `rocbudai-serve-muse`, run inside an MI355X allocation. A stock vLLM
module cannot serve Muse-Spark-Open (its model code and `muse` parsers are newer),
so the recipe builds a muse-capable vLLM from source into its
own `vllm/<ver>` Lmod module (the same `vllm_setup.sh` path the `atom` sibling
relies on), carrying the native AITER MXFP4 MoE backend for `gfx950` and layered on
a matching `pytorch` module's torch. The serve command loads `rocm/7.2.2` and that
`vllm/<ver>` module by name. The command serves the MXFP4 checkpoint
tensor-parallel across the eight GPUs with AITER flash attention, the breakable
`FULL_AND_PIECEWISE` CUDA-graph path, chunked prefill, the `muse` reasoning/tool
parsers, and DSpark speculative decoding, binding `0.0.0.0:8000` so the login node
can reach it. It advertises the model under the short name `musespark`
(`--served-model-name`), which is what the agent sends as its model id.

The recipe serves at `--max-model-len 131072` rather than the AMD recipe's default
of `32768`, because the rocBudAI persona alone is ~28k tokens; the larger window
leaves working room for the report, code, and conversation. The MXFP4 variant is
validated up to the full `1048576` context at a low `--max-num-seqs`, so the window
is a knob (`RBK_MAXLEN` / `--context-limit`), traded against concurrency.

The two halves meet on `module load`. The module reads the user's live Slurm
allocation, resolves the compute node, and sets
`ROCBUDAI_OPENAI_BASE=http://<that-node>:8000/v1` automatically. With no live
allocation it errors out and tells the user to start a server first. Detection
re-runs on every load, so after changing allocations a user reloads the module to
pick up the new node. As in the other recipes, the launcher also writes the served
context window as the model's `limit` in `opencode.json` (see
[What every recipe shares](../README.md#what-every-recipe-shares)).

## Installing the agent

Run the installer; it clones and patches rocBudAI and writes the modulefile:

```bash
./install-rocbudai-muse.sh                      # into $HOME/opt/rocbudai/muse
```

Point the serve defaults and shared cache at your own engine build (see
[Building the engine](#building-the-engine)) and, optionally, publish to a shared
prefix for the whole site:

```bash
# Variants share a parent dir: ROCBUDAI_PREFIX_ROOT sets the base (default
# $HOME/opt/rocbudai) and the installer appends the variant leaf (here `muse`).
# Override the whole path with --prefix instead if you prefer.
ROCBUDAI_PREFIX_ROOT=/shared/apps/rocbudai \
./install-rocbudai-muse.sh \
  --hf-home /shared/apps/hf-cache \
  --vllm-module vllm/0.32.1-pt2.14.1 \
  --vllm-modulepath /shared/apps/modulefiles
module use /shared/apps/rocbudai/muse/modulefiles
```

The installer prints the `module use` / `module load` lines to finish with. All
flags are in [Options](#options); `--help` lists them.

## Quick start

We start a server on a compute node, then load the module on the login node.
Below, `$PREFIX` is the install root (the installer's default is
`$HOME/opt/rocbudai/muse`; a site may stage it on a shared path instead).

Step 1: in an MI355X allocation, start the server. `--mem=0` is required, and we
use the full path because the module is not loaded on the compute node:

```bash
salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00
"$PREFIX"/bin/rocbudai-serve-muse   # wait for "Application startup complete"
```

The partition name (`mi355x`) and the GRES string (`gpu:amd_instinct_mi355_oam:8`)
are site-specific: find yours with `sinfo` and the `Gres=` line of
`scontrol show node <node>`, or set `SLURM_PARTITION` / `GPU_GRES` in `config.sh`.
`--mem=0` (whole-node memory) is the one flag that must stay, since staging the
816B checkpoint needs nearly all of the node's RAM. For a different ROCm version or
GPU, see [Porting to a different ROCm version or GPU](#porting-to-a-different-rocm-version-or-gpu).

The first start loads ~510 GB of weights and captures CUDA graphs, which takes
several minutes. It is ready when it prints `Application startup complete` and
`Uvicorn running on http://0.0.0.0:8000`.

Step 2: back on the login node, load the module from the project we want to
profile. It finds the allocation's node on its own:

```bash
module use "$PREFIX"/modulefiles
cd <your-project-dir>
module load rocbudai/muse
```

Loading the module launches the TUI exactly as every recipe does (see
[What every recipe shares](../README.md#what-every-recipe-shares)). To pin a fixed
endpoint instead of auto-detection, export
`ROCBUDAI_OPENAI_BASE=http://host:port/v1` before loading; a real remote value is
respected, a loopback one is ignored so detection can run.

## Partition and persona

- **Submit partition.** The agent dispatches GPU jobs through `rocbudai-submit` to
  `ROCBUDAI_SUBMIT_PARTITION`, baked from `--partition` at install (default
  `mi355x`). Override it per run with `-p <partition>`, or export
  `ROCBUDAI_SUBMIT_PARTITION` before `module load`. These jobs land in a separate
  allocation from the model's, which is saturated by Muse-Spark-Open.
- **Persona.** The persona matches the GPU you are *profiling*, independent of the
  GPU serving the model. Unlike `aim-engine`, this recipe pins it at install
  (`--gfx-arch`, default `gfx950`) instead of asking per session; override it for one
  shell by exporting `ROCBUDAI_GFX_ARCH=<arch>` before `module load`.

## Building the engine

The setup kit in this
directory is for reproducing the install or porting it to another MI355X system.
Every setting lives in `config.sh` and is overridable; export a variable before
sourcing to change it. By default the scripts reuse the shared prebuilt engine, so
the download and build steps are optional.

`2-build-engine.sh` builds vLLM **from the merged commit** — no container. It wraps
AMD's HPCTrainingDock `vllm_setup.sh`, which compiles vLLM (and its AITER MXFP4 MoE
kernels) against a `pytorch` Lmod module, applies the ROCm compile workarounds, and
emits a `vllm/<ver>-pt<ptver>` modulefile. The build is pinned to `VLLM_BUILD_REF`
(default: the Muse-Spark-Open merge commit; may be a branch, tag, or SHA — we
shallow-fetch it and hand it to the builder as a local branch so even a bare commit
pins exactly), labelled `VLLM_BUILD_VERSION` (`0.32.1`, which also seeds
setuptools-scm since the pinned checkout is tagless).

```bash
source ./config.sh      # override any path first, e.g. SHARE_ROOT or BASE

./1-download-muse.sh    # weights (~510 GB) + DSpark draft via a batch job; skip if reusing the shared cache
./2-build-engine.sh     # vLLM + AITER from source -> vllm module (long compile; run on a build/compute node)
./3-run-serve-test.sh   # grabs a node, serves Muse-Spark-Open, hits the API end to end
```

Prerequisite: a `pytorch` Lmod module built for your ROCm (HPCTrainingDock's
`pytorch_setup.sh` provides one; override with `PYTORCH_MODULE`, default
`pytorch/2.14.1`). On success `2-build-engine.sh` prints the two variables that
point the serve command at the module it built:

```bash
export VLLM_MODULEPATH="$VLLM_MODULEPATH"            # dir to `module use`
export VLLM_MODULE=vllm/<ver>-pt<ptver>              # exact token printed by the build
./3-run-serve-test.sh                                # or: rocbudai-serve-muse
```

We watch a run with `squeue -u $USER` and `tail -f "$LOGDIR"/*.log`; a good serve
prints the `/v1/models` listing and a `/v1/chat/completions` reply.

### Porting to a different ROCm version or GPU

The recipe is pinned and validated as one set: `rocm/7.2.2`, `pytorch/2.14.1`, and
a from-source `vllm/0.32.1-pt2.14.1` (vLLM at the Muse-Spark-Open merge commit with
its AITER MXFP4 MoE kernels), on `gfx950`. The hard rule is that **the vLLM build,
its AITER kernels, and the `pytorch` module it links must all match the ROCm they
are built against**. A mismatch does not fail cleanly; it surfaces as import/ABI
errors on the first forward pass. So move the whole set together rather than one
piece at a time.

To target a different ROCm version, override these in `config.sh` before building:

- `ROCM_MODULE`: your site's module that provides ROCm.
- `PYTORCH_MODULE`: a `pytorch` Lmod module built for that ROCm; the build links
  vLLM against its torch.
- `VLLM_BUILD_REF` / `VLLM_BUILD_VERSION`: the vLLM commit to compile and the
  version label for the emitted module.
- `AITER_RELEASE_TAG`: the AITER ref `vllm_setup.sh` builds from source alongside
  vLLM, carrying the `gfx950` MXFP4 MoE backend.

For a different GPU, also set `--gfx-arch` on the installer (persona and arch pin)
and `SLURM_PARTITION` / `GPU_GRES` for the node shape. Then run the kit
(`1-download-muse.sh`, `2-build-engine.sh`, `3-run-serve-test.sh`) and install as
usual. Only the pinned set above is tested, so expect to iterate on the vLLM commit
and the `pytorch` module on a materially different stack.

## Options

`install-rocbudai-muse.sh` bakes the agent-side defaults into the modulefile it
writes:

| Flag | Meaning |
| --- | --- |
| `--prefix DIR` | install root (default `${ROCBUDAI_PREFIX_ROOT:-$HOME/opt/rocbudai}/muse`) |
| `ROCBUDAI_PREFIX_ROOT` (env) | shared parent dir for the default prefix (default `$HOME/opt/rocbudai`); the installer appends the `muse` leaf |
| `--gfx-arch ARCH` | profiling-target persona: `gfx90a`/`mi300x`/`gfx950`/`mi300a` (default `gfx950`) |
| `--model NAME` | agent-facing served model id (default `musespark`) |
| `--context-limit N` | context window opencode advertises; keep equal to `RBK_MAXLEN` (default `131072`) |
| `--port N` | port the vLLM server listens on (default `8000`) |
| `--partition P` | Slurm partition for node auto-detect + `rocbudai-submit` (default `mi355x`) |
| `--rocm-module M` | ROCm module the modulefile loads (default `rocm/7.2.2`) |
| `--endpoint URL` | bake a fixed endpoint and skip Slurm node auto-detection |
| `--opencode-bin PATH` | opencode binary (default: first on `PATH`) |
| `--hf-home DIR` | shared weights cache, baked as module `HF_HOME` and serve default |
| `--vllm-module NAME` | `vllm/<ver>` module the serve helper loads (default `vllm/0.32.1-pt2.14.1`) |
| `--vllm-modulepath DIR` | dir to `module use` before loading `--vllm-module` |
| `--rocbudai-ref REF` / `--rocbudai-src PATH` | pin a git ref, or install from a local checkout |
| `--force` | overwrite an existing `--prefix` |
| `--dry-run` | print actions without changing anything |

The serve command is further tuned by environment variables read by
`rocbudai-serve-muse`:

| Variable | Meaning |
| --- | --- |
| `RBK_MODEL` | HF checkpoint to serve (default `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`) |
| `RBK_SERVED_NAME` | agent-facing model id (default `musespark`) |
| `RBK_PORT` | server port (default `8000`) |
| `RBK_TP` | tensor-parallel degree (default `8`, the whole node) |
| `RBK_EP` | expert parallel: `1` for throughput beyond ~32 concurrent, `0` for lowest latency (default `0`) |
| `RBK_DSPARK` | DSpark speculative decoding (default `1`; see [Turnaround](#turnaround)) |
| `RBK_DSPARK_MODEL` | draft model (default `Inferact/Muse-Spark-Open-DSpark`) |
| `RBK_MAXLEN` | context window, `--max-model-len` (default `131072`; full context is `1048576`) |
| `RBK_MAXSEQS` | `--max-num-seqs` (default `8`; raise with a lower context) |
| `RBK_REASONING_EFFORT` | server-wide default `minimal`\|`low`\|`medium`\|`high`\|`xhigh` (default `high`; see [Turnaround](#turnaround)) |
| `RBK_GPU_UTIL` | `--gpu-memory-utilization` (default `0.90`) |
| `RBK_HF_HOME` | weights cache (default the shared `hf-cache`) |
| `RBK_USE_AITER` | force AITER MXFP4 MoE on (default `1`). Required for this MXFP4 model; sets both `VLLM_ROCM_USE_AITER` and `VLLM_ROCM_USE_AITER_MOE`, overriding the `0` the vllm module defaults to. Set `0` only when porting to a non-MXFP4 checkpoint |
| `VLLM_MODULE` | Lmod `vllm/<ver>` module the serve helper loads (default `vllm/0.32.1-pt2.14.1`) |
| `VLLM_MODULEPATH` | dir to `module use` before loading `VLLM_MODULE` |
| `RBK_EXTRA` | extra args appended to the server command |

The agent side is driven by the module; the two that a user might override are:

| Variable | Meaning |
| --- | --- |
| `ROCBUDAI_OPENAI_BASE` | pin a fixed endpoint and skip node auto-detection (a loopback value is ignored) |
| `ROCBUDAI_CONTEXT_LIMIT` | context window opencode advertises; keep it equal to `RBK_MAXLEN` |

## Turnaround

Muse-Spark-Open is a reasoning model: it emits hidden reasoning before its answer.
`reasoning_effort` is `minimal|low|medium|high|xhigh` (the chat template's own
default is `medium`). The serve command pins the whole server to
`RBK_REASONING_EFFORT=high` for quality with faster turnaround (a client's own
`reasoning_effort` still wins per request). Set it to `low` for the snappiest turns,
or `xhigh` for the deepest reasoning:

```bash
RBK_REASONING_EFFORT=low rocbudai-serve-muse
```

Muse reads `reasoning_effort` as a **chat-template kwarg**, so the server-wide
default is applied with `--default-chat-template-kwargs '{"reasoning_effort":"..."}'`
(the same mechanism the ATOM/Kimi recipe uses for `thinking_effort`). vLLM merges it
into every request, and a per-request `chat_template_kwargs.reasoning_effort` wins.

The recipe's throughput lever is **DSpark speculative decoding** (`RBK_DSPARK=1`,
with the draft model `Inferact/Muse-Spark-Open-DSpark`): the draft proposes tokens
the target verifies in a batch, which AMD measured at up to +72% single-stream over
plain TP8. It is on by default; set `RBK_DSPARK=0` to disable. Note AMD validated
DSpark at `--max-model-len 32768` and the long-context window separately; if the
merged build errors with DSpark at `131072`, lower `RBK_MAXLEN` or set
`RBK_DSPARK=0`. For many concurrent
users, enable expert parallelism instead (`RBK_EP=1`), which trades a little
single-request latency for throughput beyond ~32 concurrent requests. Prefix
caching is intentionally **off** in this recipe (the AMD recipe serves with
`--no-enable-prefix-caching --enable-chunked-prefill`).

## Troubleshooting

- `'vllm' not on PATH after loading <module>`: the `vllm/<ver>` module did not
  provide vLLM. Check `./2-build-engine.sh` finished, that `VLLM_MODULE` names the
  exact token it printed, and that `VLLM_MODULEPATH` points at the parent dir the
  module lives under (the one you `module use`).
- A **segfault during `import vllm`** (crash in `triton/knobs.py` loading
  `libtriton.so`, right after torch deprecation warnings): vLLM's env_override
  promotes `libtorch_cpu.so` symbols `RTLD_GLOBAL` on ROCm, so Triton binds LLVM
  against torch's copy and crashes. `patches/vllm-rocm-symbol-promotion.patch` adds
  the `VLLM_NO_ROCM_SYMBOL_PROMOTION` opt-out and the serve command sets it; confirm
  `./2-build-engine.sh` applied the patch (`grep VLLM_NO_ROCM_SYMBOL_PROMOTION` in
  the installed `vllm/env_override.py`).
- `NotImplementedError: MuseSpark1x MXFP4 on ROCm requires the AITER_MXFP4_MXFP4
  backend; got EMULATION`: AITER MoE is off. The serve command forces it on via
  `RBK_USE_AITER=1` (which sets both `VLLM_ROCM_USE_AITER` and
  `VLLM_ROCM_USE_AITER_MOE`); make sure you did not set `RBK_USE_AITER=0`.
- `FileNotFoundError: VLLM_RUST_FRONTEND_PATH=auto but the vllm-rs binary was not
  found`: this from-source build does not compile the optional Rust frontend. The
  serve command uses the Python frontend (`VLLM_USE_RUST_FRONTEND=0`); only set it
  to `1` on a build that ships `vllm-rs`.
- A worker dies with `exitcode=-9` on load: the allocation has too little RAM.
  Re-allocate with `--mem=0` so the whole node's memory is granted.
- `no AMD GPU detected ... this looks like a login node`: the serve command only
  runs on a compute node. Start it inside an MI355X allocation.
- The TUI reports the server is not reachable: a stale loopback
  `ROCBUDAI_OPENAI_BASE` is blocking auto-detection, or no server is running.
  `module unload rocbudai/muse && module load rocbudai/muse` re-detects the node.
- `import aiter` fails on the login node: expected. The GPU libraries exist only
  on compute nodes; test inside an allocation.
- A `muse`-parser or model-arch error at startup: the `vllm/<ver>` module predates
  the Muse-Spark-Open merge. Rebuild it from a commit after vLLM PR #61879 merged
  (see the engine-availability note at the top).

## Files

Agent side (the `module load rocbudai` experience):

- `install-rocbudai-muse.sh`: the installer described above.
- `patches/rocbudai-muse.patch`: the vLLM backend patch, applied to pinned
  rocBudAI (keyed on `ROCBUDAI_BACKEND=vllm`; shared with the `atom` sibling).
- `no-gpu-addendum.md`: the instructions overlay appended when the agent host has
  no local GPU, so GPU work is dispatched via `rocbudai-submit`.

Engine side (building and serving Muse-Spark-Open):

- `config.sh`: all settings, every value overridable. Source it first.
- `1-download-muse.sh`: download the Muse-Spark-Open weights + DSpark draft via a batch job.
- `2-build-engine.sh`: build vLLM + AITER from the merged commit (via AMD's `vllm_setup.sh`); emits a `vllm/<ver>` module and applies the engine patches (`patches/vllm-*.patch`) to the installed vLLM.
- `3-run-serve-test.sh`: submit an end-to-end serve and smoke-test job.
- `rocbudai-serve-muse.sh`: the serve command, bundled so the kit is portable.
- `patches/vllm-rocm-symbol-promotion.patch`: gates vLLM's ROCm libtorch symbol promotion behind `VLLM_NO_ROCM_SYMBOL_PROMOTION` (avoids the Triton `libtriton` segfault on `import vllm`); applied to the installed vLLM by `2-build-engine.sh`.
