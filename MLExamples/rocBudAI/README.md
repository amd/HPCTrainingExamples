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

The agent and its workflow are the same in every subdirectory. What differs is the
trust boundary and the model. Neither subdirectory reproduces the cluster
deployment's guarantees. We treat them as a preview of the agent, and follow the
rocBudAI repository's `INSTALL.md` for a deployment that must keep code private.

- [`ollama`](ollama/README.md): runs a small model (Laguna XS) with an ollama
  daemon we start ourselves on a GPU node. It needs no Slurm settings, and no
  airgap is enforced.
- [`aim-engine`](aim-engine/README.md): runs the agent as a thin client of a large
  model served by [`LLM_Serving/aim-engine`](../LLM_Serving/aim-engine/README.md),
  on a cluster we assert is trusted and in-boundary. The agent then also runs from
  a login node or a laptop without a local GPU.
