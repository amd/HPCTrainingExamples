# Adversarial Multi-Agent ICF Capsule Optimizer

A multi-agent system that autonomously optimizes Inertial Confinement Fusion (ICF) capsule designs under realistic manufacturing uncertainty. A Design Explorer agent maximizes fusion yield while a Defect Adversary agent stress-tests designs with manufacturing imperfections, orchestrated by a Principal Investigator agent — all powered by LLM reasoning and a fast physics surrogate. After a campaign runs, the same Principal Investigator can be **interactively queried** about the results: it answers analysis questions ("why did the best design win?", "how robust is it to defects?") from the recorded run data, with multi-turn memory across the conversation.

## Architecture

```
┌─────────────────────────────────────────────┐
│              AutoGen Agent Swarm            │
│                                             │
│   ┌───────────────────────────────────┐     │
│   │    Principal Investigator (PI)    │     │
│   │       Orchestrator / Evaluator    │     │
│   └──────┬────────────────┬──────────┘      │
│          │                │                 │
│   ┌──────▼──────┐  ┌─────▼────────┐         │
│   │   Design    │  │    Defect    │         │
│   │  Explorer   │  │  Adversary   │         │
│   └─────────────┘  └──────────────┘         │
│          │                                  │
│   ┌──────▼──────┐                           │
│   │ User Proxy  │ ◄── tool execution only   │
│   └──────┬──────┘                           │
└──────────┼──────────────────────────────────┘
           │ stdio / JSON-RPC
    ┌──────▼──────────────┐
    │   MCP Server        │
    │  (ICF Surrogate)    │
    └─────────────────────┘
```

The agents communicate through AutoGen's group chat with constrained speaker transitions. Only the PI can invoke the physics simulation, ensuring a structured optimization loop. The physics engine is decoupled from the agents via the Model Context Protocol (MCP), allowing the backend to be swapped from this fast surrogate to a full radiation hydrodynamics code without changing any agent logic.

Every campaign's structured results (best design, all sampled designs and their yields, mean/σ statistics, sweep curves, rendered media) are retained in a session-scoped store. A dedicated **conversational PI agent** (no tool access, separate from the orchestrating PI) draws on this store to answer follow-up analysis questions, so the user can interrogate what was run without launching a new simulation. The router automatically distinguishes a *request to run* (e.g. "optimize…", "sweep…", "show a movie…") from a *question about prior results*, sending the latter to the debrief agent.

The system runs against either backend (selected by `ICF_API_TYPE`):

- **`anthropic`** — a frontier cloud model. The PI invokes the simulation through AutoGen's native tool-calling.
- **`openai`** — a local, OpenAI-compatible endpoint (vLLM serving the fine-tuned `gptoss-20b-hedp` weights). Because smaller open-weight models are unreliable at formal tool-calling, in this mode the PI instead emits a plain-text `RUN_SIMULATION: R0=..., v0=..., ...` line, which a hook parses and executes against the MCP server, injecting the results back into the conversation. This mode also uses deterministic speaker scheduling, response sanitization (stripping stray tool-call/null tokens), and bounded conversation history so long campaigns stay within the model's context window.

## Physics Model

The surrogate implements a 0D deceleration-phase model combining:

- **Hurricane et al. (2014)** — dual-piston stagnation mechanics with mode-1 asymmetry
- **Betti et al. (2002)** — Spitzer electron thermal ablation and marginal ignition physics
- **NRL Plasma Formulary** — Gamow DT thermonuclear burn rate
- **Linear Rayleigh-Taylor** instability growth tracking

The model solves a stiff 9-component ODE system (shell positions, velocities, hot-spot energy, yield, RT growth, fuel mass, hot-spot mass) using the Radau implicit integrator. Each evaluation takes < 1 second, enabling thousands of design iterations.

### Optimization Formulation

The system solves a min-max game:

```
max_x  min_d  Y(x, d)
```

where **x** = (R0, v0, T0, M_sh, M_hs) are design parameters and **d** = (delta, mode, roughness) are manufacturing defects, subject to a kinetic energy budget KE = ½ M_sh v0² = 0.015 × E_laser. The laser energy E_laser is parsed from the user's request (any value, not a fixed 3 MJ), and the resulting budget is computed in code and handed to the agents — so a request for, say, a 5 MJ laser produces a 75 kJ budget and the agents scale the design accordingly.

## Files

| File | Description |
|---|---|
| `app.py` | Multi-agent orchestration, request routing (single/sweep/media/analysis), conversational-PI debrief, session result store, Gradio UI, LLM configuration |
| `icf_core.py` | Standalone physics engine (`solve_implosion` / `compute_metrics` + CLI) |
| `icf_viz.py` | Media generation: implosion movie (ffmpeg) and diagnostic time-history plots |
| `icf_mcp_server.py` | MCP tool server wrapping the physics engine with parameter validation |
| `start_app.sh` | Launch script for the agent application |
| `start_vllm.sh` | Launch script for local vLLM inference server |
| `requirements.txt` | Python dependencies |
| `docs/summary.tex` | Manuscript (IEEE format) |

## Setup

### Prerequisites

- Python 3.10+
- For local inference: AMD GPUs with ROCm, podman or docker, vLLM container image

Before starting any container, make sure to prep the underlying fs (especially if running rootless):
```
cat > ~/.config/containers/storage.conf
[storage]
driver = "overlay"

[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"

[storage.options.overlay]
```

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

The system supports two LLM backends: a cloud API (Anthropic) and local inference (vLLM with OpenAI-compatible API).

### Option A: Cloud API (Anthropic)

Set the API credentials and run:

```bash
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

ICF_API_TYPE=anthropic ./start_app.sh
```

### Option B: Local Inference (vLLM + ROCm)

Download the latest ROCm vLLM dev image from [dockerhub](http://hub.docker.com/r/rocm/vllm-dev/tags), e.g.
```
podman pull rocm/vllm-dev:nightly_main_20260531 && podman save rocm/vllm-dev:nightly_main_20260531 -o vllm_rocm_nightly_main_20260531.tar
```
Start the vLLM server in one terminal, then the app in another:

```bash
# Terminal 1 — start vLLM
./start_vllm.sh

# Terminal 2 — start the app
./start_app.sh
```

The app opens a Gradio web UI with a public share link (no X server needed). The interface uses the Origin theme (dark mode by default), shows the AMD logo, streams the agent conversation in a chat panel, and renders plots and movies in side panels. Agent messages stream to the browser in real time as each agent contributes.

### Usage Modes

The request is routed by keyword into one of four modes:

**1. Single design campaign** (default) — one full agent optimization at a given laser energy.

> Find a robust capsule design that maximizes fusion yield at 3 MJ laser energy

**2. Laser-energy sweep** (triggered by "sweep", "scan", "as a function of", "vs laser energy", "curve"). Runs one full campaign per energy point and plots **mean fusion yield with ±1σ error bars** (the spread of yields the agents explore at each energy, reflecting sensitivity to manufacturing defects). The energy points are parsed from the request, supporting **multiple range segments each with their own step size**:

> Sweep yield vs laser energy from 1.0 to 2.0 MJ in increments of 0.1 MJ and 2.0 to 10.0 MJ in increments of 1.0 MJ

Also accepts a point count ("...from 1 to 10 MJ with 6 points") or a single uniform step.

**3. Simulation media** (triggered by "movie", "animation", "render", "show the implosion", "diagnostic", etc.). Runs a campaign, takes the best design found, and renders an **implosion movie** (density field through stagnation) plus **diagnostic time-history plots** (radius, areal density, temperature).

> Optimize a design at 4 MJ and show me a movie of the implosion

**4. Analysis / debrief** (triggered once at least one campaign has run, by any question that is not itself a run request — e.g. it ends in "?" or uses words like "why", "how", "compare", "explain", "robust"). Instead of starting a new simulation, the question is routed to a conversational Principal Investigator that reasons over all results collected this session and replies in the chat panel. The exchange has multi-turn memory, so follow-up questions retain context.

> Why did the best design win, and how robust is it to the defects we tried?
> Which single defect parameter hurt the yield the most?
> Compare the 3 MJ and 5 MJ campaigns.

This mode is purely interpretive: the debrief agent has no simulation tool and never alters the design — it only analyzes the recorded data (best design, every sampled design and its yield, mean/σ, and any sweep curve). To start fresh work, use an explicit run phrasing ("optimize…", "sweep…", "render a movie…").

Each campaign is capped at a fixed number of agent rounds (`SWEEP_MAX_ROUNDS` / `SINGLE_MAX_ROUNDS` / `MEDIA_MAX_ROUNDS` in `app.py`) and also stops early when the yield plateaus (improvement < 5% for two consecutive rounds). If a campaign produces no valid simulation, a physically reasonable fallback design at the target energy keeps the sweep/media output populated.

### Running the Physics Engine Standalone

```bash
source venv/bin/activate

# default parameters (symmetric, no defects)
python3 icf_core.py

# with defects
python3 icf_core.py --delta 0.1 --mode 20 --roughness 2e-6

# with plotting
python3 icf_core.py --plot

# custom design point
python3 icf_core.py --R0 1.2e-4 --v0 -5e5 --M_sh 3.6e-7 --M_hs 8e-9 --T0 1.2
```

## Environment Variables

### Backend Selection

| Variable | Default | Description |
|---|---|---|
| `ICF_API_TYPE` | `anthropic` | LLM backend: `anthropic` or `openai` |

### Per-Role LLM Configuration

Each agent role (Explorer, Frontier) has independent settings with fallback defaults:

| Variable | Anthropic Default | OpenAI Default | Description |
|---|---|---|---|
| `ICF_EXPLORER_API_KEY` | `$ANTHROPIC_API_KEY` | `unused` | Explorer LLM API key |
| `ICF_EXPLORER_BASE_URL` | `$ANTHROPIC_BASE_URL` | `http://localhost:8000/v1` | Explorer LLM endpoint |
| `ICF_EXPLORER_MODEL` | `Claude-Sonnet-4.5` | `gptoss-20b-hedp` | Explorer model name |
| `ICF_FRONTIER_API_KEY` | `$ANTHROPIC_API_KEY` | `unused` | PI + Adversary API key |
| `ICF_FRONTIER_BASE_URL` | `$ANTHROPIC_BASE_URL` | `http://localhost:8000/v1` | PI + Adversary endpoint |
| `ICF_FRONTIER_MODEL` | `Claude-Sonnet-4.5` | `gptoss-20b-hedp` | PI + Adversary model name |
| `ICF_EXPLORER_CUSTOM_HEADERS` | `$ANTHROPIC_CUSTOM_HEADERS` | — | Custom HTTP headers (Anthropic mode only) |
| `ICF_FRONTIER_CUSTOM_HEADERS` | `$ANTHROPIC_CUSTOM_HEADERS` | — | Custom HTTP headers (Anthropic mode only) |

### vLLM Server Configuration

| Variable | Default | Description |
|---|---|---|
| `ICF_VLLM_MODEL_PATH` | `~/hfmodels/gptoss-20b-hedp.03302026` | Path to model weights |
| `ICF_VLLM_MODEL_NAME` | `gptoss-20b-hedp` | Served model name |
| `ICF_VLLM_IMAGE_TAR` | `~/vllm_rocm_nightly_main_20260531.tar` | vLLM container image tarball |
| `ICF_VLLM_PORT` | `8000` | Port for the vLLM server |
| `ICF_VLLM_TP_SIZE` | `1` | Tensor parallel degree (number of GPUs) |
| `ICF_VLLM_MAX_MODEL_LEN` | `16384` | Maximum sequence length |

## Parameter Bounds

The MCP server enforces hard physical bounds on all inputs. Values outside these ranges are clamped and a warning is returned to the agent:

| Parameter | Min | Max | Description |
|---|---|---|---|
| R0 | 3×10⁻⁵ m | 5×10⁻⁴ m | Deceleration-phase shell radius |
| v0 | -8×10⁵ m/s | -1×10⁵ m/s | Implosion velocity (inward) |
| T0 | 0.1 keV | 10 keV | Initial hot-spot temperature |
| M_sh | 10⁻⁸ kg | 2×10⁻⁶ kg | Shell mass |
| M_hs | 10⁻⁹ kg | 10⁻⁷ kg | Initial hot-spot mass |
| delta | 0.0 | 0.15 | Mode-1 asymmetry fraction |
| mode | 2 | 40 | RT perturbation mode number |
| roughness | 0.0 m | 3×10⁻⁶ m | Surface roughness |

## References

- O. A. Hurricane et al., "Fuel gain exceeding unity in an inertially confined fusion implosion," *Nature*, 506, 343–348, 2014.
- R. Betti et al., "Deceleration phase of inertial confinement fusion implosions," *Phys. Plasmas*, 9(5), 2277–2286, 2002.
- J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory, 2019.
