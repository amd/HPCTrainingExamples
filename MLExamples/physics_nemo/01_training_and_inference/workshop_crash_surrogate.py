"""Train a toy crash surrogate and evaluate the supplied checkpoint.

See README.md for setup, results, and the artifact layout.
"""
import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys, os, time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, help="New run directory (relative to the caller's directory)")
parser.add_argument("--epochs", type=int, help="Toy training epochs (default: YAML configuration)")
parser.add_argument("--warmup", type=int, default=3, help="Warmup forwards per simulation")
parser.add_argument("--repeats", type=int, default=10, help="Timed forwards per simulation")
args = parser.parse_args()
if args.warmup < 1 or args.repeats < 1 or (args.epochs is not None and args.epochs < 1):
    parser.error("epochs, warmup, and repeats must be positive")
ROOT = Path(__file__).resolve().parent
OUTPUT = (args.output_dir.resolve() if args.output_dir else
          ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
OUTPUT.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT)
os.environ["WORKSHOP_VTP_OUTPUT_DIR"] = str(OUTPUT / "vtp")
os.environ.setdefault("MPLBACKEND", "Agg")
T0 = time.time()
def _log(msg): print(f"[T+{time.time()-T0:6.1f}s] {msg}", flush=True)

import sys, os
sys.path.insert(0, os.path.abspath("src"))

import torch
import physicsnemo
import pyvista as pv
import hydra, omegaconf

print("torch:", torch.__version__)
print("cuda/HIP device available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
print("physicsnemo:", physicsnemo.__version__)
print("pyvista:", pv.__version__)
print("hydra-core:", hydra.__version__, "| omegaconf:", omegaconf.__version__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
metadata = {
    "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(), "platform": platform.platform(),
    "torch": torch.__version__, "rocm": torch.version.hip,
    "physicsnemo": physicsnemo.__version__, "pyvista": pv.__version__,
    "device": str(DEVICE),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "force_adam_mi300a": bool(os.environ.get("FORCE_ADAM_MI300A")),
}
(OUTPUT / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Results:", OUTPUT)

# `bq_torch_patch` forces PhysicsNeMo's radius_search onto the pure-torch backend.
# On ROCm, warp-lang imports successfully but has no HIP backend and crashes at
# runtime — this patch must be imported before any model is built. Harmless if
# warp happens to work fine on your hardware (e.g. real CUDA).
_log('cell 3: importing bq_torch_patch...')
import bq_torch_patch
_log('cell 3: done')

_log('cell 5: reading mesh...')
mesh = pv.read("data/vtp_train/run1.vtp")
print("n_points (mesh nodes):", mesh.n_points)
print("n_cells:", mesh.n_cells)

disp_fields = sorted([k for k in mesh.point_data.keys() if k.startswith("displacement_t")])
print(f"\n{len(disp_fields)} displacement timesteps, e.g.:", disp_fields[:3], "...", disp_fields[-1])

strain_fields = sorted([k for k in mesh.cell_data.keys() if "plastic_strain" in k])
stress_fields = sorted([k for k in mesh.cell_data.keys() if "stress_vm" in k])
print(f"{len(strain_fields)} plastic-strain timesteps, {len(stress_fields)} stress timesteps (per-cell)")
_log('cell 5: done')

_log('cell 6: plotting mesh (matplotlib)...')
# Plot the mesh at t0 and at the final timestep, colored by displacement magnitude.
# Uses matplotlib (not pyvista's VTK renderer) so this works headless without
# a GPU/EGL/OSMesa OpenGL context -- pyvista's off_screen renderer needs one of
# those and isn't guaranteed to be present in a minimal container/notebook env.
import numpy as np
import matplotlib.pyplot as plt

coords0 = mesh.points
final_disp = np.asarray(mesh.point_data[disp_fields[-1]])
coords_final = coords0 + final_disp
disp_mag = np.linalg.norm(final_disp, axis=-1)

# The raw VTP bundles a few points that are not part of the crushable
# structure and would otherwise show up as disconnected stray dots:
#  (a) ~150 duplicate connector/weld nodes with no cell in this file's
#      polygon list at all (isolated in the point-adjacency graph), and
#  (b) a small (~48-point) rigid-wall/fixture mesh with its own cells but
#      exactly zero displacement at every timestep -- present, but not part
#      of the deforming assembly.
# The bumper beam and its end brackets are assembled from several parts that
# share no mesh nodes with each other (joined by welds, not shared vertices),
# so they appear as separate connected components too -- keep all of those;
# only drop components with no cells at all, or that never move.
parent = np.arange(mesh.n_points)
def _find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra != rb:
        parent[ra] = rb
for i in range(mesh.n_cells):
    pids = mesh.get_cell(i).point_ids
    for p in pids[1:]:
        _union(pids[0], p)
roots = np.array([_find(i) for i in range(mesh.n_points)])
root_ids, root_sizes = np.unique(roots, return_counts=True)
max_disp_all = np.zeros(mesh.n_points)
for k in disp_fields:
    np.maximum(max_disp_all, np.linalg.norm(np.asarray(mesh.point_data[k]), axis=-1), out=max_disp_all)
moving_roots = {r for r in root_ids
                if root_sizes[root_ids == r][0] > 1 and max_disp_all[roots == r].max() > 1.0}
structure_mask = np.array([r in moving_roots for r in roots])
n_dropped = mesh.n_points - structure_mask.sum()
if n_dropped:
    print(f"Excluding {n_dropped} point(s) not part of the deforming structure "
          f"(unconnected connector nodes / static fixture geometry) from the mesh plot.")
structure_idx = np.where(structure_mask)[0]

# Subsample points for a readable scatter (full mesh is ~14k nodes)
rng = np.random.default_rng(0)
idx = rng.choice(structure_idx, size=min(4000, len(structure_idx)), replace=False)

# The beam is long (~2 m) but thin (~150 mm), so matplotlib's default cubic
# 3D box would squash it into a misleading blocky shape. set_box_aspect with
# the actual per-axis data range keeps proportions true to the real geometry.
import matplotlib.ticker as mticker
def _style_3d_ax(ax, pts, title):
    ranges = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-6)
    ax.set_box_aspect(tuple(ranges))
    ax.set_title(title, pad=-4, fontsize=11)
    ax.set_xlabel("x (mm)", labelpad=-2, fontsize=8)
    ax.set_ylabel("y (mm)", labelpad=-2, fontsize=8)
    ax.set_zlabel("z (mm)", labelpad=-6, fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.view_init(elev=16, azim=-58)
    ax.tick_params(axis="both", labelsize=7, pad=-2)

fig = plt.figure(figsize=(11, 3.6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(coords0[idx, 0], coords0[idx, 1], coords0[idx, 2], s=2, c="gray")
_style_3d_ax(ax1, coords0[idx], "t=0 (undeformed)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
sc = ax2.scatter(coords_final[idx, 0], coords_final[idx, 1], coords_final[idx, 2],
                  s=2, c=disp_mag[idx], cmap="inferno")
_style_3d_ax(ax2, coords_final[idx], "t=0.5s (crushed), colored by |displacement|")
fig.colorbar(sc, ax=ax2, shrink=0.7, pad=0.1, label="displacement magnitude (mm)")

fig.savefig(OUTPUT / "mesh.png", dpi=150, bbox_inches="tight")
plt.close(fig)
_log('cell 6: done')

from omegaconf import OmegaConf
from hydra.utils import instantiate

toy_cfg = OmegaConf.load("checkpoint/toy_config.yaml")
if args.epochs is not None:
    toy_cfg.training.epochs = args.epochs
toy_cfg.datapipe.stats_dir = str(OUTPUT / "toy_stats")
OmegaConf.save(toy_cfg, OUTPUT / "toy_config.yaml", resolve=True)
print(OmegaConf.to_yaml(toy_cfg, resolve=True))

_log('cell 9: instantiating reader + toy dataset...')
reader = instantiate(toy_cfg.reader)

train_dataset = instantiate(
    toy_cfg.datapipe,
    name="toy_train",
    reader=reader,
    split="train",  # "train" computes and saves fresh normalization stats
    logger=None,
)
print(f"Train dataset: {train_dataset.num_samples} samples")

sample0 = train_dataset[0]
print(sample0)
print("\nglobal_features (design variables + load case) for sample 0:")
for k, v in sample0.global_features.items():
    print(f"  {k}: {v.item():.3f}")
_log('cell 9: done')

_log('cell 12: instantiating model + optimizer...')
from utils import combined_crash_loss, build_muon_optimizer

model = instantiate(toy_cfg.model).to(DEVICE)
model.train()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

data_stats = dict(
    node={k: v.to(DEVICE) for k, v in train_dataset.node_stats.items()},
    edge={},
    feature={k: v.to(DEVICE) for k, v in getattr(train_dataset, "feature_stats", {}).items()},
    target={k: v.to(DEVICE) for k, v in getattr(train_dataset, "target_stats", {}).items()},
)
target_scale = data_stats["target"].get("target_scale", torch.zeros(0, device=DEVICE))
channel_std = torch.cat([torch.ones(3, device=DEVICE), target_scale]) if target_scale.numel() > 0 else None

import os
if os.environ.get("FORCE_ADAM_MI300A"):
    print("FORCE_ADAM_MI300A set -- using Adam (Muon's Newton-Schulz bf16 GEMMs hit a known hipBLASLt/MI300A-228CU tuning gap, see ROCm/rocm-systems#4084)")
    optimizer = torch.optim.Adam(model.parameters(), lr=toy_cfg.training.start_lr)
else:
    try:
        optimizer = build_muon_optimizer(model, toy_cfg)
        print("Using Muon optimizer (matches the real training run)")
    except ImportError:
        print("Muon unavailable (needs torch>=2.9) -- falling back to Adam for this toy loop")
        optimizer = torch.optim.Adam(model.parameters(), lr=toy_cfg.training.start_lr)
_log('cell 12: done')
metadata["optimizer"] = type(optimizer).__name__

_log('cell 13: starting toy training loop...')
import time

EPOCHS = toy_cfg.training.epochs
losses = []
t0 = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(train_dataset.num_samples):
        sample = train_dataset[i].to(DEVICE)
        optimizer.zero_grad()
        pred = model(sample=sample, data_stats=data_stats)
        loss = combined_crash_loss(
            pred,
            sample.node_target,
            sample.node_features["coords"],
            data_stats["node"]["pos_std"],
            alpha=toy_cfg.training.displacement_loss_alpha,
            peak_loss_weight=toy_cfg.training.peak_loss_weight,
            peak_loss_top_k=toy_cfg.training.peak_loss_top_k,
            channel_std=channel_std,
            peak_plastic_strain_loss_weight=toy_cfg.training.peak_plastic_strain_loss_weight,
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / train_dataset.num_samples
    losses.append(avg_loss)
    print(f"epoch {epoch+1:3d}/{EPOCHS}  avg_loss={avg_loss:.4f}  elapsed={time.time()-t0:.1f}s")

print("\nDone. This is a mechanics demo, not a converged surrogate -- see part 2 for the real result.")
metadata["toy_training_seconds"] = time.time() - t0
with (OUTPUT / "training_loss.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["epoch", "average_training_loss"])
    writer.writerows(enumerate(losses, start=1))
_log('cell 13: done')

_log('cell 14: plotting loss curve...')
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(losses) + 1), losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("avg training loss")
plt.title(f"Toy training loop ({train_dataset.num_samples} samples)")
plt.grid(alpha=0.3)
plt.savefig(OUTPUT / "training_loss.png", dpi=150)
plt.close()
_log('cell 14: done')

_log('cell 16: instantiating real_model + loading checkpoint...')
print("\nPart 2: real model size, inference speed, and error against two simulations")
# Release the toy training state before measuring the independent real model.
del model, optimizer, pred, loss, sample, sample0, data_stats, train_dataset
import gc
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
from physicsnemo.utils import load_checkpoint
from inference_measurements import model_sizes, measure_inference

real_cfg = OmegaConf.load("checkpoint/config.yaml")
OmegaConf.save(real_cfg, OUTPUT / "evaluation_config.yaml", resolve=True)

real_model = instantiate(real_cfg.model).to(DEVICE).eval()
epoch_loaded = load_checkpoint("checkpoint/checkpoints", models=real_model, device=DEVICE)
print(f"Loaded real checkpoint at epoch {epoch_loaded}")
print(f"Model parameters: {sum(p.numel() for p in real_model.parameters()):,}")
metadata.update(model_sizes(real_model, "checkpoint/checkpoints"))
print(f"Real model archive: {metadata['model_archive_gb']:.6f} GB (decimal, weights archive only)")
print(f"Model parameters and buffers: {metadata['model_tensor_gb']:.6f} GB (excludes runtime activations)")
_log('cell 16: done')

_log('cell 17: instantiating real test dataset...')
# split="test" LOADS the existing stats from checkpoint/stats/ (computed on the
# real 225-run dataset) instead of recomputing them from this small VTP set --
# this is the same pattern used by surrogate_opt/predictor_2d_thickness.py.
test_reader = instantiate(real_cfg.reader)
test_dataset = instantiate(
    real_cfg.datapipe,
    name="real_eval",
    reader=test_reader,
    split="test",
    logger=None,
)
print(f"Evaluation dataset: {test_dataset.num_samples} samples (run19, run201; excluded from toy training, included in checkpoint training)")

real_data_stats = dict(
    node={k: v.to(DEVICE) for k, v in test_dataset.node_stats.items()},
    edge={},
    feature={k: v.to(DEVICE) for k, v in getattr(test_dataset, "feature_stats", {}).items()},
)
_log('cell 17: done')

_log('cell 18: running inference on evaluation runs...')
# Compare surrogate prediction vs. ground truth on each evaluation run,
# for all three channels: peak intrusion (displacement), peak plastic
# strain, peak von Mises stress.
run_names = ["run19", "run201"]
metrics = []
timings = []

print(f"{'run':<8}{'peak_disp_pred':>16}{'peak_disp_true':>16}{'disp_rel%':>11}"
      f"{'strain_pred':>13}{'strain_true':>13}{'strain_rel%':>13}"
      f"{'stress_pred':>13}{'stress_true':>13}{'stress_rel%':>13}")

with torch.no_grad():
    for idx, run_name in enumerate(run_names):
        sample = test_dataset[idx].to(DEVICE)
        pred, timing = measure_inference(
            real_model, sample, real_data_stats, DEVICE,
            warmup=args.warmup, repeats=args.repeats,
        )  # [N, T, 5]: one full trajectory
        timings.append(dict(run=run_name, **timing))
        target = sample.node_target
        coords0 = sample.node_features["coords"]
        pos_std = test_dataset.node_stats["pos_std"].to(DEVICE)

        pred_disp_mm = (pred[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(1, 1, -1)
        true_disp_mm = (target[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(1, 1, -1)
        pred_peak_disp = torch.linalg.norm(pred_disp_mm, dim=-1).max().item()
        true_peak_disp = torch.linalg.norm(true_disp_mm, dim=-1).max().item()
        disp_rel = 100.0 * abs(pred_peak_disp - true_peak_disp) / true_peak_disp

        pred_peak_strain = pred[:, :, 3].max().item()
        true_peak_strain = target[:, :, 3].max().item()
        strain_rel = 100.0 * abs(pred_peak_strain - true_peak_strain) / true_peak_strain

        pred_peak_stress = pred[:, :, 4].max().item()
        true_peak_stress = target[:, :, 4].max().item()
        stress_rel = 100.0 * abs(pred_peak_stress - true_peak_stress) / true_peak_stress
        metrics.append(dict(
            run=run_name, peak_displacement_pred_mm=pred_peak_disp,
            peak_displacement_true_mm=true_peak_disp, displacement_relative_error_pct=disp_rel,
            peak_strain_pred=pred_peak_strain, peak_strain_true=true_peak_strain,
            strain_relative_error_pct=strain_rel,
            peak_stress_pred=pred_peak_stress, peak_stress_true=true_peak_stress,
            stress_relative_error_pct=stress_rel,
        ))

        print(f"{run_name:<8}{pred_peak_disp:>16.1f}{true_peak_disp:>16.1f}{disp_rel:>11.2f}"
              f"{pred_peak_strain:>13.4f}{true_peak_strain:>13.4f}{strain_rel:>13.2f}"
              f"{pred_peak_stress:>13.4f}{true_peak_stress:>13.4f}{stress_rel:>13.2f}")
        print(f"  {run_name}: median inference {timing['median_inference_ms']:.3f} ms, "
              f"{timing['trajectories_per_second']:.2f} full trajectories/s "
              f"({args.warmup} warmups, {args.repeats} timed forwards)")
_log('cell 18: done')
with (OUTPUT / "evaluation.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
metadata.update(status="complete", checkpoint_epoch=epoch_loaded, total_seconds=time.time() - T0)
(OUTPUT / "inference_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
(OUTPUT / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Saved artifacts to", OUTPUT)
