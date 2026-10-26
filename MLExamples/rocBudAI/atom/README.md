# rocBudAI over ATOM, on Kimi-K3

| At a glance | |
| --- | --- |
| Model | Kimi-K3, frontier (~1.5 TB weights) |
| Engine | ATOM (vLLM-like) on an MI355X node |
| Needs | an 8-GPU MI355X via Slurm (`--mem=0`), the 1.5 TB weight cache, no local GPU |
| Setup | a few minutes; first server start loads the weights |

This directory runs [rocBudAI](../README.md) as a thin client of
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) served on the same cluster by
[ATOM](https://github.com/ROCm/ATOM), AMD's vLLM-like, OpenAI-compatible engine.
The experience is stock rocBudAI, unchanged (see
[What every recipe shares](../README.md#what-every-recipe-shares)); only the model
moves. Where the stock deployment serves a curated model with ollama behind a
Slurm gate, here Kimi-K3 runs on an MI355X compute node and the agent runs from the
cluster's login node, so it never competes with the model for CPU and needs no GPU
of its own.

Where [`aim-engine`](../aim-engine/README.md) serves its model on a Kubernetes
cluster through AIM Engine, this recipe serves it directly on a Slurm compute node
with ATOM. Everything stays on one HPC system: the agent on the login node, the
model on a compute node, and the weights in a shared offline cache.

## Requirements and setup time

> This setup does not reproduce the stock rocBudAI airgap and Slurm gating. The
> model, the weights cache and the agent all stay on the cluster, and the module
> disables opencode's phone-home, but nothing here blocks the login node's own
> egress. Privacy rests on the cluster being trusted and in-boundary. It is a
> preview, not the production airgap (see
> [What every recipe shares](../README.md#what-every-recipe-shares)).

We need the following before starting:

- An AMD Instinct MI355X node (`gfx950`), eight GPUs, reachable through Slurm. We
  must request the whole node's memory: `--exclusive` grants all GPUs but not RAM
  on this cluster, and Kimi-K3 (about 1.5 TB of weights) is OOM-killed on load
  without `--mem=0`. The serve command refuses a low-memory allocation and prints
  the fix.
- A login node that shares the compute nodes' filesystem, where the agent runs.
- The Kimi-K3 weights in a shared, offline HuggingFace cache (about 1.5 TB). The
  cache is downloaded once and reused by everyone; see
  [Building the engine](#building-the-engine).
- The ATOM engine and a matching AITER wheel staged on a shared path. On this
  cluster both are already installed and world-readable, so most users skip the
  build entirely.
- `opencode` on `PATH`. The module ships a pinned copy; a site that stages its own
  points `ROCBUDAI_OPENCODE_BIN` at it.

On this cluster the runtime is already installed and world-readable, so there is
nothing to build: we load the module and start a server (see
[Quick start](#quick-start)). On a fresh cluster, staging the engine takes a few
minutes plus the one-time weights download, which for Kimi-K3's 1.5 TB is bound by
the site's bandwidth and storage.

## How it works

The recipe has two halves: the agent side (a modulefile any user loads) and the
engine side (the ATOM server each user starts in their own allocation).

The agent side is produced by `install-rocbudai-atom.sh`, which, like its
[`aim-engine`](../aim-engine/README.md) sibling, does four things:

1. it clones rocBudAI at a pinned ref;
2. it applies a small vLLM/ATOM backend patch (`patches/rocbudai-atom.patch`);
3. it drops in the no-GPU addendum and the serve helper;
4. it writes a modulefile (Lmod `.lua` or classic Tcl, auto-detected) that reuses
   rocBudAI's own load hook.

It installs into a prefix in our home directory by default and touches nothing
system-wide. The patch keys everything on `ROCBUDAI_BACKEND=vllm` and leaves the
ollama path untouched: loading the module points the stock launcher at an
OpenAI-compatible endpoint, declares Kimi-K3's context window so opencode
auto-compacts instead of overflowing, and launches the TUI. On this cluster the
agent is already installed at a world-readable shared prefix, so most users skip
the install and just `module load`.

Because the agent runs on a login node with no local GPU, the no-GPU addendum
engages and dispatches each profiler job through `rocbudai-submit` (see
[What every recipe shares](../README.md#what-every-recipe-shares)). One point is
specific here: the model's own allocation is saturated by Kimi-K3, so dispatched
GPU jobs land in a separate allocation on the `mi355x` partition
(`ROCBUDAI_SUBMIT_PARTITION`).

The server is `rocbudai-serve-kimi`, run inside an MI355X allocation. The cluster's
`vllm/0.19.1` module cannot serve Kimi-K3's architecture, and the vLLM that can
needs a newer ROCm than the site's `rocm/7.2.2`. ATOM targets exactly this ROCm
and GPU and installs as pure Python, reusing the `vllm` module's torch 2.10. ATOM
needs a newer AITER than that module ships (the first run died on
`ImportError: cannot import name 'topk_select' from 'aiter'`), so a pinned AITER
release wheel (`v0.1.25`, built for `rocm7.2`) is layered ahead of the module on
`PYTHONPATH`, where the first entry wins. The command serves the MXFP4 MoE
checkpoint with online `ptpc_fp8` quantization and an fp8 KV cache, tensor-parallel
across the eight GPUs, binding `0.0.0.0:8000` so the login node can reach it.

Mixing a newer AITER and ATOM against the `vllm` module's torch 2.10 means their
Triton-facing kernels do not all match the module's Triton, so two of them are
pinned to version-stable paths:

- ATOM's packed host-to-device helper calls a cached Triton kernel directly, an
  ABI that the module's Triton rejects (`function takes exactly 14 arguments
  (13 given)`). A one-line patch (`patches/atom-packed-h2d.patch`) routes it
  through Triton's normal JIT launcher instead; the build applies it automatically.
- AITER's Kimi Delta Attention runs through an experimental Gluon kernel that uses
  a Gluon API the module's Triton lacks (`'SwizzledSharedLayout' object has no
  attribute '_flatten_ir'`). The serve command sets `AITER_FDA_USE_GLUON=0`, which
  selects AITER's pure-Triton KDA kernel.

Because these depend on the exact engine versions, the recipe pins them: ATOM at
commit `67c54da4e` and AITER at `v0.1.25`, both validated against `rocm/7.2.2` and
`vllm/0.19.1-pt2.10.0` on `gfx950`.

The two halves meet on `module load`. The module reads the user's live Slurm
allocation, resolves the compute node, and sets
`ROCBUDAI_OPENAI_BASE=http://<that-node>:8000/v1` automatically. With no live
allocation it errors out and tells the user to start a server first. Detection
re-runs on every load, so after changing allocations a user reloads the module to
pick up the new node. As in the other recipes, the launcher also writes Kimi-K3's
served context window as the model's `limit` in `opencode.json` (see
[What every recipe shares](../README.md#what-every-recipe-shares)).

## Installing the agent

Run the installer; it clones and patches rocBudAI and writes the modulefile:

```bash
./install-rocbudai-atom.sh                      # into $HOME/rocbudai-atom
```

If your site has already staged the agent at a shared prefix, skip the install and
`module use <that-prefix>/modulefiles` directly.

Point the serve defaults and shared cache at your own engine build (see
[Building the engine](#building-the-engine)) and, optionally, publish to a shared
prefix for the whole site:

```bash
./install-rocbudai-atom.sh \
  --prefix /shared/apps/rocbudai-atom \
  --hf-home /shared/apps/hf-cache \
  --atom-pylib /shared/apps/rocbudai-atom/atom-pylib \
  --aiter /shared/apps/rocbudai-atom/aiter-wheel
module use /shared/apps/rocbudai-atom/modulefiles
```

The installer prints the `module use` / `module load` lines to finish with. All
flags are in [Options](#options); `--help` lists them.

## Quick start

We start a server on a compute node, then load the module on the login node.
Below, `$PREFIX` is the install root (the installer's default is
`$HOME/rocbudai-atom`; a site may stage it on a shared path instead).

Step 1: in an MI355X allocation, start the server. `--mem=0` is required, and we
use the full path because the module is not loaded on the compute node:

```bash
salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00
"$PREFIX"/bin/rocbudai-serve-kimi   # wait for "Application startup complete"
```

The partition name (`mi355x`) and the GRES string (`gpu:amd_instinct_mi355_oam:8`)
are site-specific: find yours with `sinfo` and the `Gres=` line of
`scontrol show node <node>`, or set `SLURM_PARTITION` / `GPU_GRES` in `config.sh`.
`--mem=0` (whole-node memory) is the one flag that must stay, since Kimi-K3 needs
nearly all of the node's RAM to load. For a different ROCm version or GPU, see
[Porting to a different ROCm version or GPU](#porting-to-a-different-rocm-version-or-gpu).

The first start loads 1.5 TB of weights and builds JIT kernels, which takes
several minutes. It is ready when it prints `Application startup complete` and
`Uvicorn running on http://0.0.0.0:8000`.

Step 2: back on the login node, load the module from the project we want to
profile. It finds the allocation's node on its own:

```bash
module use "$PREFIX"/modulefiles
cd <your-project-dir>
module load rocbudai/atom
```

Loading the module launches the TUI exactly as every recipe does (see
[What every recipe shares](../README.md#what-every-recipe-shares)). To pin a fixed
endpoint instead of auto-detection, export
`ROCBUDAI_OPENAI_BASE=http://host:port/v1` before loading; a real remote value is
respected, a loopback one is ignored so detection can run.

## Building the engine

Most users on this cluster never build anything. The setup kit in this directory
is for reproducing the install or porting it to another MI355X system. Every
setting lives in `config.sh` and is overridable; export a variable before sourcing
to change it. By default the scripts reuse the shared prebuilt engine, so the
download and build steps are optional.

`2-build-engine.sh` checks out the pinned ATOM commit (`ATOM_COMMIT` in
`config.sh`), applies every patch in `patches/` to it (idempotent, so re-runs are
safe), then installs ATOM and the pinned AITER wheel into layer dirs. To track a
newer ATOM, override `ATOM_COMMIT`; the patches may then need refreshing.

```bash
source ./config.sh      # override any path first, e.g. SHARE_ROOT or BASE

./1-download-kimi.sh    # weights (about 1.5 TB) via a batch job; skip if reusing the shared cache
./2-build-engine.sh     # ATOM plus the pinned AITER wheel into layer dirs (the core step)
./3-run-serve-test.sh   # grabs a node, serves Kimi-K3, hits the API end to end
```

If we build our own engine, `2-build-engine.sh` prints the two variables that
point the serve command at it:

```bash
RBK_ATOM_PYLIB="$ATOM_PYLIB" RBK_AITER="$AITER_DIR" "$SERVE"
```

The one rule that matters when porting: on `PYTHONPATH`, the AITER wheel must come
before `atom-pylib` and the `vllm` module so the newer `aiter` wins. The serve
command does this for us. We watch a run with `squeue -u $USER` and
`tail -f "$LOGDIR"/*.log`; a good serve prints the `/v1/models` listing and a
`/v1/chat/completions` reply.

### Porting to a different ROCm version or GPU

The recipe is pinned and validated as one set: `rocm/7.2.2`,
`vllm/0.19.1-pt2.10.0`, AITER `v0.1.25` (built for `rocm7.2`), and ATOM
`67c54da4e`, on `gfx950`. The hard rule is that **ATOM and the AITER wheel must
match the ROCm, PyTorch, and Triton that the serving modules provide**. A mismatch
does not fail cleanly; it surfaces as the Triton errors in
[Troubleshooting](#troubleshooting) (the `packed_h2d` "14 arguments" crash and the
Gluon `SwizzledSharedLayout` crash are both AITER/ATOM-vs-module-Triton skew). So
move the whole set together rather than one piece at a time.

To target a different ROCm version, override these in `config.sh` before building:

- `ROCM_MODULE` and `VLLM_MODULE`: your site's modules that provide ROCm and a
  matching PyTorch + Triton (the engine reuses the `vllm` module's torch).
- `AITER_ROCM`, `AITER_RELEASE_TAG` (and `AITER_MANIFEST` for a nightly): an AITER
  wheel built for that ROCm. Its ABI must match the module's PyTorch/Triton.
- `ATOM_COMMIT`: a commit compatible with that PyTorch/Triton. The bundled
  `patches/atom-packed-h2d.patch` targets the pinned commit, so a different commit
  may need the patch refreshed (or may already include the fix).

For a different GPU, also set `--gfx-arch` on the installer (persona and arch pin)
and `SLURM_PARTITION` / `GPU_GRES` for the node shape. Then run the kit
(`1-download-kimi.sh`, `2-build-engine.sh`, `3-run-serve-test.sh`) and install as
usual. Only the pinned set above is tested, so expect to iterate on the AITER/ATOM
versions and patches on a materially different stack.

## Options

`install-rocbudai-atom.sh` bakes the agent-side defaults into the modulefile it
writes:

| Flag | Meaning |
| --- | --- |
| `--prefix DIR` | install root (default `$HOME/rocbudai-atom`) |
| `--gfx-arch ARCH` | profiling-target persona: `gfx90a`/`mi300x`/`gfx950`/`mi300a` (default `gfx950`) |
| `--model NAME` | served model id (default `moonshotai/Kimi-K3`) |
| `--context-limit N` | context window opencode advertises; keep equal to `RBK_MAXLEN` (default `131072`) |
| `--port N` | port the ATOM server listens on (default `8000`) |
| `--partition P` | Slurm partition for node auto-detect + `rocbudai-submit` (default `mi355x`) |
| `--rocm-module M` | ROCm module the modulefile loads (default `rocm/7.2.2`) |
| `--endpoint URL` | bake a fixed endpoint and skip Slurm node auto-detection |
| `--opencode-bin PATH` | opencode binary (default: first on `PATH`) |
| `--hf-home DIR` | shared weights cache, baked as module `HF_HOME` and serve default |
| `--atom-pylib DIR` / `--aiter DIR` | engine layer dirs baked into the serve helper |
| `--rocbudai-ref REF` / `--rocbudai-src PATH` | pin a git ref, or install from a local checkout |
| `--force` | overwrite an existing `--prefix` |
| `--dry-run` | print actions without changing anything |

The serve command and the module are further tuned by environment variables. The
serve knobs are read by `rocbudai-serve-kimi`:

| Variable | Meaning |
| --- | --- |
| `RBK_MODEL` | model to serve (default `moonshotai/Kimi-K3`) |
| `RBK_PORT` | server port (default `8000`) |
| `RBK_TP` | tensor-parallel degree (default `8`, the whole node) |
| `RBK_MAXLEN` | context window, `--max-model-len` (default `131072`; Kimi-K3 supports up to 1048576) |
| `RBK_REASONING_EFFORT` | server-wide default thinking effort `low`\|`high`\|`max` (default `high`; see [Turnaround](#turnaround)) |
| `RBK_PREFIX_CACHING` | reuse the KV of a shared prompt prefix across turns (default `1`; see [Turnaround](#turnaround)) |
| `RBK_GPU_UTIL` | `--gpu-memory-utilization` (default `0.93`) |
| `RBK_HF_HOME` | weights cache (default the shared `hf-cache`) |
| `RBK_ATOM_PYLIB` | ATOM layer dir, for a private build |
| `RBK_AITER` | AITER wheel layer dir, for a private build |
| `RBK_EXTRA` | extra args appended to the server command |

The agent side is driven by the module; the two that a user might override are:

| Variable | Meaning |
| --- | --- |
| `ROCBUDAI_OPENAI_BASE` | pin a fixed endpoint and skip node auto-detection (a loopback value is ignored) |
| `ROCBUDAI_CONTEXT_LIMIT` | context window opencode advertises; keep it equal to `RBK_MAXLEN` |

## Turnaround

Kimi-K3 is a reasoning model: it always emits a hidden chain-of-thought
(`reasoning_content`) before its answer, and it defaults to `reasoning_effort`
`"max"`. On an agent turn that means a long silent "thinking" pause — at `"max"`
a single turn generated ~9,000 characters of reasoning (~48 s at ~65 tok/s),
versus ~10 s at `"high"` or `"low"` with no loss of answer quality for
interactive coding/profiling. Decode throughput itself is healthy (warm TTFT
~0.2 s, ~65 tok/s single-stream, all 8 GPUs saturated during generation); the
turnaround is almost entirely reasoning-token volume.

The serve command therefore defaults the whole server to `RBK_REASONING_EFFORT=high`
(passed as `--default-chat-template-kwargs '{"thinking_effort":"high"}'`, which
every request inherits unless the client sends its own `reasoning_effort`). Set it
to `low` for the snappiest turns or `max` to restore the model's benchmark setting:

```bash
RBK_REASONING_EFFORT=low rocbudai-serve-kimi
```

A client can also override per request by sending a top-level `reasoning_effort`
field; `"none"` disables thinking entirely for that request.

The other half of turnaround is prefill. Each agent turn resends a large stable
prefix (the persona, the no-GPU addendum, and the growing conversation history),
so the serve command enables prefix caching (`RBK_PREFIX_CACHING=1`, ATOM's
default): the KV of that shared prefix is reused across turns and only the new
tokens are prefilled, cutting per-turn prefill from several seconds (measured
~4.5 s at ~50k tokens without it) to near zero. Decode itself is ~65 tok/s with
all 8 GPUs saturated. Set `RBK_PREFIX_CACHING=0` to disable.

## Troubleshooting

- `cannot import name 'topk_select'`: the module's older AITER is winning. Confirm
  `python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))"` points at
  the AITER wheel layer, meaning it is first on `PYTHONPATH`.
- `function takes exactly 14 arguments (13 given)` (a ModelRunner dies after
  startup): ATOM's packed host-to-device Triton ABI mismatch. The build applies
  `patches/atom-packed-h2d.patch`; if you built your own ATOM, confirm the patch
  landed (`git -C "$BASE/ATOM" apply --reverse --check patches/atom-packed-h2d.patch`).
- `'SwizzledSharedLayout' object has no attribute '_flatten_ir'` (a ModelRunner
  dies on the first prompt): AITER's Gluon KDA kernel against the module's Triton.
  The serve command sets `AITER_FDA_USE_GLUON=0`; make sure it is in effect
  (`AITER_FDA_USE_GLUON=0 rocbudai-serve-kimi`).
- A ModelRunner dies with `exitcode=-9` on load: the allocation has too little
  RAM. Re-allocate with `--mem=0` so the whole node's memory is granted.
- `no AMD GPU detected ... this looks like a login node`: the serve command only
  runs on a compute node. Start it inside an MI355X allocation.
- The TUI reports the server is not reachable: a stale loopback
  `ROCBUDAI_OPENAI_BASE` is blocking auto-detection, or no server is running.
  `module unload rocbudai/atom && module load rocbudai/atom` re-detects the node.
- `import aiter` fails on the login node: expected. The GPU libraries exist only
  on compute nodes; test inside an allocation.

## Files

Agent side (the `module load rocbudai` experience):

- `install-rocbudai-atom.sh`: the installer described above.
- `patches/rocbudai-atom.patch`: the vLLM/ATOM backend patch, applied to pinned
  rocBudAI (keyed on `ROCBUDAI_BACKEND=vllm`).
- `no-gpu-addendum.md`: the instructions overlay appended when the agent host has
  no local GPU, so GPU work is dispatched via `rocbudai-submit`.

Engine side (building and serving Kimi-K3):

- `config.sh`: all settings, every value overridable. Source it first.
- `1-download-kimi.sh`: download the Kimi-K3 weights through a batch job.
- `2-build-engine.sh`: install ATOM and the pinned AITER wheel into layer dirs.
- `3-run-serve-test.sh`: submit an end-to-end serve and smoke-test job.
- `rocbudai-serve-kimi.sh`: the serve command, bundled so the kit is portable.
- `patches/atom-packed-h2d.patch`: the Triton-ABI fix applied to the pinned ATOM
  commit during the build.
