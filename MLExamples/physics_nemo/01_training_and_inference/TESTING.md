# Validation and troubleshooting

Use the [shared environment setup](../README.md#shared-environment) and the
[current exercise commands](README.md#setup-and-run). The old exercise-local
installers have been removed. Transfer the complete `physics_nemo` directory
when moving the workshop to another machine, so the shared installer is included.

## Check the current environment

Inside the ROCm 10 PyTorch container on an allocated GPU node:

```bash
"$HOME/venvs/physicsnemo-rocm10/bin/python3" -c "import torch, physicsnemo, scipy; print(torch.__version__); print(physicsnemo.__version__); print(scipy.__version__); print('GPU:', torch.cuda.is_available())"
```

Run the first exercise using the README command. Check that the toy loss decreases,
the supplied checkpoint loads, and the exercise writes `run.json`, `evaluation.csv`,
and `inference_timing.json`.
Confirm `run.json` reports `status: complete`, and compare the accuracy metrics with
the [recorded results](README.md#results-and-artifacts). Timing depends on hardware
and software; do not treat historical timings as targets for every system.

- Start Apptainer with `--rocm` and check GPU visibility inside the container.
- Set `RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500`
  for single-process checkpoint loading under Slurm.
- Use `FORCE_ADAM_MI300A=1` for the toy loop on MI300A. Historical Muon runs hit
  very slow optimizer steps; Adam completed a 30-epoch run in roughly two minutes
  on one tested node.
- Use the script for live progress and saved artifacts; notebook execution tools
  can buffer output until completion.
- Launch Jupyter from this exercise directory using the shared environment's Python
  and select a kernel from that environment.

The notes below preserve earlier measurements and the reason for the compatibility
patch. They do not imply that every later installer or hardware combination has
been tested.

## Historical ROCm 10 checks: torch.cdist compute_mode needed a workaround (fixed)

Running `workshop_crash_surrogate.py` under a ROCm 10.0.0 environment
(the former wheel-based installer) originally crashed on the first forward
pass:

```
File ".../physicsnemo/nn/functional/neighbors/radius_search/_torch_impl.py", line 106
    dists = torch.cdist(points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
torch.AcceleratorError: CUDA error: invalid configuration argument (hipErrorInvalidConfiguration)
```

This comes from `BQWarp`'s ball query (used for GeoTransolver's multi-scale
local geometric features, `include_local_features: true`), which calls
PhysicsNeMo's `radius_search` — forced onto the pure-torch backend by
`src/bq_torch_patch.py` since `warp-lang` has no HIP backend on ROCm.

**Investigation, using standalone reproducers (`N` swept from 4096 to
16384, `torch.cdist(x, x, p=2.0,
compute_mode="donot_use_mm_for_euclid_dist")` on a `torch.ones(1, N, 3,
device="cuda")` tensor):**

- On ROCm 10.0.0 and 7.14.1, this exact call raised
  `hipErrorInvalidConfiguration` at every size we tested — including the
  workshop's real ~13,882-node mesh. Torch version didn't change this
  (2.10.0/2.11.0/2.13.0 all behaved the same way).
- On ROCm 7.2.x (7.2.1/7.2.4 — the version this workshop's shipped
  checkpoint was originally trained/verified on), the call didn't raise,
  but the returned distances didn't match a CPU float64 reference for a
  large fraction of pairs (frequently exactly `0.0`). Checked against the
  real mesh's (normalized) coordinates: e.g. reference distance
  `3.017...`, this specific GPU compute_mode returned `0.0`; 82% of all
  192.7M pairs in the 13,882×13,882 distance matrix disagreed with the
  reference by more than 1.0 (units: normalized coord space). The
  identical call on CPU, and the identical call on GPU with the *default*
  (unset) `compute_mode`, both matched the reference — only the explicit
  `donot_use_mm_for_euclid_dist` GPU path disagreed in our testing.
- The obvious alternative, forcing `compute_mode="use_mm_for_euclid_dist_if_necessary"`
  (the matmul/Gram-trick), didn't pan out as a fix either: benchmarked
  head-to-head on the real mesh, it was **~35x slower** (571ms vs 16ms/call)
  *and* its values disagreed with the reference for pairs within the
  actual ball-query radius (consistent with catastrophic cancellation in
  `|x|²+|y|²−2xy`) — consistent with PhysicsNeMo's own code comment already
  flagging the unset-`compute_mode` default as "numerically unstable" for
  this use case.

**Fix (`src/bq_torch_patch.py`):** replace `torch.cdist` globally (for the
process) with a manual broadcast-based distance
(`(x.unsqueeze(-2)-y.unsqueeze(-3)).pow(2).sum(-1).sqrt()`), a different
code path that doesn't go through any cdist compute_mode kernel. Patched
globally rather than scoped to just `radius_search`'s two call sites since
this is about `torch.cdist` itself in the environments we tested, not
something specific to how PhysicsNeMo calls it — other `cdist` call sites
elsewhere in physicsnemo (`lagrangian_dataset.py`, diffusion
preconditioners, `figconvnet`, `mesh_lsq_gradient.py`) would see the same
behavior if ever exercised with this compute_mode on this hardware, even
though this workshop doesn't currently touch those paths.

Verified on real MI300A hardware:

- Matches the CPU float64 reference exactly on the cases we checked (the
  native `donot_use_mm` kernel path did not, per above).
- No exception at any tested size (4096–16384) on ROCm 10.0.0.
- ~3% forward-pass overhead relative to the native `donot_use_mm` kernel on
  ROCm 7.2.4 (6.43ms vs 6.22ms at N=13882) — not a meaningful cost, and it's
  called only a handful of times per forward pass (once per radius scale ×
  per context/local-feature path), not on the model's dominant compute path.
- **End-to-end, no retraining needed:** ran the full
  `workshop_crash_surrogate.py` on ROCm 10.0.0 (torch
  2.11.0+rocm10.0.0) — toy training loop (30 epochs, full local features)
  converged normally (loss 3.43 → 0.19), and the *existing* 200-epoch
  pretrained checkpoint (trained on ROCm 7.2.1, never retrained) loaded and
  evaluated on the 2 held-out runs with results matching this doc's
  previously-recorded baseline:

  | run | peak disp rel err | strain rel err | stress rel err |
  |---|---|---|---|
  | run19 | 0.57% | 0.75% | 11.59% |
  | run201 | 0.26% | 6.49% | 12.55% |

  (0.57% matches the "~0.5%" baseline recorded in the original verification notes, run on
  the original ROCm 7.2.1 setup — i.e. the fix preserves accuracy exactly,
  and the pretrained checkpoint's weights hold up fine despite the
  compute_mode discrepancy described above during its original training
  run.)

Two paths were considered and rejected in favor of this patch:
- **Retrain with a different architecture** (`TransolverOneShot` /
  `MeshGraphNetOneShot`, both already implemented in `src/rollout.py`)
  — neither touches `cdist`/`radius_search`/`BQWarp` at all, so this would
  also work, but requires retraining from scratch and (per the upstream
  `NVIDIA/physicsnemo` crash example's own benchmarks) is less accurate
  than GeoTransolver on this exact dataset. Not needed once the patch was
  shown to fully preserve the existing checkpoint's accuracy.

**Cross-checked on a second, independent torch+ROCm10 build**
(the former container package installer): rather than building
torch from wheels on top of a bare ROCm base image, this variant pulls
AMD's official prebuilt image
(`docker://rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0`),
which ships torch 2.11.0+rocm10.0.0 already installed. The same
`bq_torch_patch.py` fix applies unchanged, and a full
`workshop_crash_surrogate.py` run produced the same result: toy training
converged normally, and the existing checkpoint evaluated with the same
0.57%/0.26% peak-displacement error as the wheel-based installation —
confirming this isn't specific to one particular torch build/install
method. That historical installer used an external package directory. Current setup
uses the shared home-directory environment described in the parent README.
- **Force `compute_mode="use_mm_for_euclid_dist_if_necessary"`** — ruled
  out above (both wrong and 35x slower).
