# rocBudAI as a user, with ollama

| At a glance | |
| --- | --- |
| Model | Laguna XS, small |
| Engine | ollama, self-run on a GPU node |
| Needs | 1 AMD Instinct GPU (CDNA2+) with ROCm, ~25 GB disk, no admin |
| Setup | ~10 min |

This directory installs [rocBudAI](https://github.com/AMD-HPC/rocBudAI) into our
home directory and runs it against an ollama daemon that we start ourselves on a
GPU node. Nothing needs an administrator: there is no Slurm comment or partition
to request, no allow-list to configure, and no system service. The agent itself
is the stock one, with its personas, discovery interview and benchmarking tools.

## Requirements and setup time

> No airgap is enforced in this setup. Nothing blocks network traffic from the
> node, so the agent and the tools it runs can reach the internet. Do not use it on
> code or data that must stay private. This setup is only for experiencing
> rocBudAI: a production deployment must follow the stock rocBudAI install in
> [`INSTALL.md`](https://github.com/AMD-HPC/rocBudAI/blob/main/INSTALL.md) to
> get the airgap, the Slurm gating and the curated models.

We repeat this message wherever a session shows its state. The launcher prints it
at startup and the welcome banner states it. `rocbudai-doctor` warns about it,
and `rocbudai-airgap-check` fails and exits with status 1.

Beyond that, we need the following before starting:

- A GPU node with at least one AMD Instinct GPU of the CDNA2 generation or
  newer, for example an MI210, MI250 or MI300X, and ROCm (for example
  `module load rocm`). The default model uses about 22 GB of GPU memory at full
  context, which every such GPU has.
- An interactive shell on that node, with network access to GitHub and
  ollama.com for the install and the model pull. On a Slurm system this must be
  a shell inside our allocation: the installer refuses to run on a login node.
- About 25 GB of disk: 20 GB for the model, plus 2.5 GB when the installer
  downloads ollama. Ollama stores models in `~/.ollama/models` by default;
  `--models-dir` puts them on a larger file system.
- `git`, `curl` and `sha256sum`, plus `zstd` when the installer downloads ollama.
  Laguna's tool calling needs ollama 0.32.3 or newer.

Once we hold the node, setup takes about ten minutes. On an MI300X node with a
fast network, the install took 40 s including the ollama download, and pulling
the model took one minute. The first agent turn then arrived within two and a
half minutes of running `rocbudai`. Downloads scale with the site's bandwidth,
and we have not timed CDNA2 GPUs, where the first turn is slower. Without a GPU
node, the [`aim-engine`](../aim-engine/README.md) example runs the agent with no
local GPU, but it needs a Kubernetes cluster that already runs AIM Engine.

## How it works

`install-rocbudai-ollama.sh` clones rocBudAI at a pinned commit and applies
`patches/rocbudai-ollama.patch`. It stages checksum-verified `opencode` and `rg`
binaries. It uses the ollama on `PATH`, or downloads the pinned release into the
prefix when there is none. An ollama too old for Laguna (see
[Requirements and setup time](#requirements-and-setup-time)) stops the install
unless we pass `--download-ollama`. Finally it writes a `rocbudai/ollama`
modulefile (Lmod or Tcl) under the install prefix.

The patch adds a user mode, keyed on `ROCBUDAI_USER_MODE=1`, which the modulefile
sets. In user mode the launcher skips the Slurm, partition and allow-list gates.
When the daemon or the model is missing, it prints the user-level command to fix
it instead of pointing at the site administrator. Everything after the gates is
upstream code running unchanged. The modulefile also sets `OLLAMA_NO_CLOUD=1`,
which turns off ollama's remote inference and web search, and `OLLAMA_NOPRUNE=1`,
as upstream does. These limit what ollama itself does but are not an airgap.

The default model is `laguna-xs-2.1:q4_K_M`. We apply the overlay
in `Modelfile.laguna-xs-2.1-q4_K_M.rocbudai` to it, as upstream does for Laguna S:
it adds rocBudAI's hard rules as a system prompt and sets `num_thread 1`, without
which ollama leaves the GPU idle for this model. The modulefile asks ollama for
the full 262144-token context with flash attention and a `q8_0` key/value cache.

## Quick start

We work in a shell on the GPU node throughout. On a Slurm system we get one with,
for example, `srun -N1 --gres=gpu:1 -t 01:00:00 --pty bash`. A bare `salloc`
shell often stays on the login node, and the installer checks for that. Install
once:

```bash
./install-rocbudai-ollama.sh --models-dir /path/with/space/ollama-models
```

Then load the module and start our own ollama daemon. rocBudAI
reserves the last visible GPU for benchmarks, so we start the daemon on the GPUs
the topology helper assigns to the model:

```bash
module use $HOME/rocbudai-ollama/modulefiles
module load rocbudai/ollama
( d=$(bash "$ROCBUDAI_ROOT/libexec/rocbudai-detect-topology.sh" | sed -n 's/^ROCBUDAI_LLM_DEVICES=//p')
  [ -n "$d" ] && export ROCR_VISIBLE_DEVICES=$d
  exec ollama serve ) > ~/ollama-rocbudai.log 2>&1 &
```

Ollama has no authentication, so on a node shared with other users anyone on
that node can reach this daemon and use our model. We prefer an exclusive
allocation, or at least a non-default port set with `--ollama-host` at install
time.

Pull the model and apply the overlay under the same name:

```bash
ollama pull laguna-xs-2.1:q4_K_M
ollama create laguna-xs-2.1:q4_K_M -f $HOME/rocbudai-ollama/deploy/ollama-models/Modelfile.laguna-xs-2.1-q4_K_M.rocbudai
```

Then start a session from the project we want to profile:

```bash
cd <your-project-dir>
rocbudai
```

Unlike the cluster module, loading `rocbudai/ollama` only sets the environment, and
`rocbudai` starts the agent. `rocbudai --continue` resumes the last session in the
directory, and `rocbudai --new` forces a fresh one. Session files (`AGENTS.md`,
`opencode.json`, `.rocbudai-*`) are written into the project directory, so add
them to its `.gitignore`. When we are done, we stop the daemon with
`pkill -u $USER -x ollama`, which also releases its GPU.

## Options

| Flag | Meaning |
| --- | --- |
| `--prefix DIR` | install root (default `$HOME/rocbudai-ollama`) |
| `--model NAME` | ollama model the agent runs (default `laguna-xs-2.1:q4_K_M`) |
| `--ollama-host H:P` | address of our ollama daemon (default `127.0.0.1:11434`) |
| `--models-dir DIR` | where ollama stores models (baked as `OLLAMA_MODELS`) |
| `--gfx-arch ARCH` | pin the persona: `gfx90a`, `mi300x`, `gfx950` or `mi300a` |
| `--ollama-bin PATH` | use this ollama binary instead of the one on `PATH` |
| `--download-ollama` | download the pinned ollama into the prefix even if one is on `PATH` |
| `--rocbudai-ref REF` | git ref to pin (default: the ref the patch targets) |
| `--rocbudai-src PATH` | use a local rocBudAI checkout instead of cloning |
| `--force` | overwrite an existing `--prefix` |
| `--allow-no-allocation` | skip the Slurm allocation check, for a compute node reached another way (for example SSH into our allocation) |
| `--dry-run` | print actions without changing anything |
| `--help` | usage |

The launcher picks the persona from the GPU that `rocminfo` reports. Some MI300X
systems do not report the MI300X marketing name, and the launcher then falls back
to the MI300A persona; on those we install with `--gfx-arch mi300x`. Exporting
`ROCBUDAI_MODEL` before loading the module overrides the model for one shell.
A model other than Laguna needs its own entry in rocBudAI's opencode templates
to get a correct context limit.

`rocbudai-submit` dispatches GPU work through `sbatch`, so it works only on a
system with Slurm. Everything else runs on the node where we start `rocbudai`.

## Files

- `install-rocbudai-ollama.sh`: the installer described above.
- `patches/rocbudai-ollama.patch`: the user-mode branch for the launcher,
  `rocbudai-doctor` and `rocbudai-airgap-check`, plus the Laguna XS context limit
  in the opencode templates.
- `Modelfile.laguna-xs-2.1-q4_K_M.rocbudai`: the ollama overlay for the default
  model.
