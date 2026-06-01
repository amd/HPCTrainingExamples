# Adversarial Multi-Agent ICF Capsule Optimizer

A multi-agent system that autonomously optimizes Inertial Confinement Fusion (ICF) capsule designs under realistic manufacturing uncertainty. A Design Explorer agent maximizes fusion yield while a Defect Adversary agent stress-tests designs with manufacturing imperfections, orchestrated by a Principal Investigator agent — all powered by LLM reasoning and a fast physics surrogate.

## Architecture

```
┌─────────────────────────────────────────────┐
│              AutoGen Agent Swarm            │
│                                             │
│   ┌───────────────────────────────────┐     │
│   │    Principal Investigator (PI)    │     │
│   │       Orchestrator / Evaluator    │     │
│   └──────┬────────────────┬──────────┘     │
│          │                │                 │
│   ┌──────▼──────┐  ┌─────▼────────┐        │
│   │   Design    │  │    Defect    │        │
│   │  Explorer   │  │  Adversary   │        │
│   └─────────────┘  └──────────────┘        │
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

where **x** = (R0, v0, T0, M_sh, M_hs) are design parameters and **d** = (delta, mode, roughness) are manufacturing defects, subject to a fixed kinetic energy budget: KE = ½ M_sh v0² = 0.015 × E_laser.

## Files

| File | Description |
|---|---|
| `app.py` | Multi-agent orchestration, Gradio UI, LLM configuration |
| `icf_core.py` | Standalone physics engine (ODE solver) |
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

The app opens a Gradio web UI with a public share link (no X server needed). Enter a prompt like:

> Find a robust capsule design that maximizes fusion yield at 3 MJ laser energy

Agent messages stream to the browser in real time as each agent contributes.

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
| `ICF_VLLM_MAX_MODEL_LEN` | `4096` | Maximum sequence length |

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
