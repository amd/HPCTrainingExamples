# rocBudAI

[rocBudAI](https://github.com/AMD-HPC/rocBudAI) is AMD-HPC's ROCm profiling
assistant. It is an [opencode](https://opencode.ai) agent that runs from a terminal
and is driven by GPU-architecture personas. Through a short discovery interview, it
walks us through profiling and optimizing our own code with the ROCm profilers.

The rocBudAI repository tracks the development of the agent and its system-level
deployment on a cluster. An administrator installs that deployment, which
provides:

- a shared module and an ollama service;
- Slurm gating of the GPU allocations the agent runs in;
- an airgap that keeps code and prompts inside the cluster.

This directory shows how to try rocBudAI at a user level instead, without
administrator access. Each subdirectory installs the stock agent into our home
directory and changes only where the model is served.

The agent and its workflow are identical across the three subdirectories; only the
trust boundary and the model differ. The behavior they share is collected in
[What every recipe shares](#what-every-recipe-shares) below, so each subdirectory
documents only what is unique to it.

- [`ollama`](ollama/README.md): runs a small model (Laguna XS) with an ollama
  daemon we start ourselves on a GPU node. It needs no Slurm settings, and no
  airgap is enforced.
- [`aim-engine`](aim-engine/README.md): runs the agent as a thin client of a large
  model served by [`LLM_Serving/aim-engine`](../LLM_Serving/aim-engine/README.md),
  on a cluster we assert is trusted and in-boundary. The agent then also runs from
  a login node or a laptop without a local GPU.
- [`atom`](atom/README.md): runs the agent against a frontier model (Kimi-K3)
  served on the same cluster by AMD's [ATOM](https://github.com/ROCm/ATOM) engine
  on an MI355X node. The model runs on a compute node and the agent on the login
  node, so like `aim-engine` it needs no local GPU, but model, weights and agent
  all stay on one HPC system.

## What every recipe shares

The three recipes differ only in the model and where it is served; the agent and
its workflow are the same. The common parts are described once here, and each
subdirectory links back rather than repeating them.

Stock experience. The agent is unmodified rocBudAI with the same terminal UI: a
session picker when the project directory holds prior sessions, otherwise the
welcome banner and the seven-question discovery interview. `rocbudai --continue`
resumes the last session in the directory and `rocbudai --new` forces a fresh one;
session naming, the idle auto-nudge and the resume hints behave as upstream. How
that UI is launched is the one thing that varies, and each recipe says so.

Session files. A session writes `AGENTS.md`, `opencode.json` and `.rocbudai-*`
into the project directory we launch from, so we add those to its `.gitignore`.

Running without a local GPU. When the agent runs where there is no GPU, a login
node or a laptop, it cannot profile locally. The recipe then appends
`no-gpu-addendum.md` to opencode's instructions, overriding the persona: the agent
dispatches every build, run and profiler job through `rocbudai-submit` to Slurm
instead of running it locally. On a real GPU host the addendum is omitted and the
stock local flow applies.

Context window. A remotely served model reaches opencode through a custom provider
it has no metadata for, so the launcher declares the served context window as that
model's `limit` in `opencode.json`. That is what lets opencode auto-compact the
history and cap output tokens instead of sending a request the engine rejects.

Not the production airgap. None of these reproduce the deployment's enforced
airgap and Slurm gating; they are previews for trying rocBudAI. A deployment that
must keep code private follows the stock install in
[`INSTALL.md`](https://github.com/AMD-HPC/rocBudAI/blob/main/INSTALL.md). Each
recipe states its own trust boundary in its own README.
