# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch

from physicsnemo.optim import CombinedOptimizer


def load_global_features(json_path: str) -> dict[str, dict[str, float]]:
    """
    Load global features JSON once.

    Returns:
        dict[str, dict[str, float]]:
            Mapping run_id -> global feature dict
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Global features file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Global features JSON must be a dict keyed by run_id")

    # Optional: sanity check values
    for run_id, features in data.items():
        if not isinstance(features, dict):
            raise TypeError(f"Global features for run '{run_id}' must be a dict")

    return data


def get_global_features_for_run(
    all_global_features: dict[str, dict[str, float]],
    run_id: str,
) -> dict[str, float]:
    """
    Fetch global features for a single run.

    Args:
        all_global_features: output of load_global_features
        run_id: key identifying the run (e.g. derived from filename)

    Returns:
        dict[str, float]: global scalar features for this run
    """
    try:
        return all_global_features[run_id]
    except KeyError:
        raise KeyError(f"run_id '{run_id}' not found in global features file")


def displacement_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords0: torch.Tensor,
    pos_std: torch.Tensor,
    alpha: float = 2.0,
    eps: float = 1e-8,
    channel_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MSE weighted per (node, timestep) by ground-truth displacement magnitude, so
    states near peak intrusion contribute more to the loss than plain averaging
    would give them.

    Args:
        pred, target: [N, T, Fo] with the first 3 channels of Fo = position.
        coords0: [N, 3] normalized t0 coordinates.
        pos_std: [3] per-coordinate std used to denormalize position into
            physical units before computing displacement magnitude.
        alpha: weight strength; weight = 1 + alpha * mag / mean(mag).
        eps: numerical stability for the mean-normalization.
        channel_std: optional [Fo] per-output-channel std used to normalize
            each channel's squared error before the displacement-derived
            weight is applied. Without this, non-displacement channels (e.g.
            effective_plastic_strain, stress_vm) are weighted purely by
            displacement magnitude with no accounting for their own scale —
            since these channels live on very different numeric scales from
            displacement and from each other, that leaves their contribution
            to the loss essentially arbitrary. None preserves prior behavior
            exactly (backward compatible).
    """
    target_disp_norm = target[:, :, :3] - coords0.unsqueeze(1)  # [N,T,3]
    target_disp_mm = target_disp_norm * pos_std.view(1, 1, -1)
    mag = torch.linalg.norm(target_disp_mm, dim=-1)  # [N,T]

    weight = 1.0 + alpha * mag / (mag.mean() + eps)  # [N,T]
    sq_error = torch.square(pred - target)  # [N,T,Fo]
    if channel_std is not None:
        sq_error = sq_error / (channel_std.view(1, 1, -1) ** 2 + eps)
    return torch.mean(sq_error * weight.unsqueeze(-1))


def peak_intrusion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords0: torch.Tensor,
    pos_std: torch.Tensor,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Auxiliary loss term that directly penalizes displacement-magnitude error at
    the actual peak-intrusion location(s) — the region the surrogate is
    validated/optimized against downstream.

    An earlier version of this term compared a location-agnostic soft-max over
    ALL predicted nodes against the true global max. That let the optimizer
    "satisfy" the term by raising displacement anywhere in the field, with no
    gradient tying it to the specific node/timestep where peak intrusion
    actually occurs — it made held-out peak-intrusion error WORSE in practice
    (see PHYSICSNEMO_2x_manual retrain history, +peak-loss run vs baseline).

    This version instead identifies the top_k (node, timestep) pairs by GROUND
    TRUTH displacement magnitude (i.e. the true peak-crush region) and computes
    a direct MSE between predicted and true magnitude at exactly those
    locations. Because the locations are fixed by the ground truth (not
    differentiated through), gradients flow straight to the nodes that matter
    instead of being redistributed by a soft-max across the whole mesh.

    Args:
        pred, target: [N, T, Fo] with the first 3 channels of Fo = position.
        coords0: [N, 3] normalized t0 coordinates.
        pos_std: [3] per-coordinate std used to denormalize position into
            physical units (mm) before computing displacement magnitude. The
            returned loss is rescaled back into normalized units (divided by
            |pos_std|) so it's comparable in magnitude to the base
            displacement_weighted_mse term computed on normalized pred/target.
        top_k: number of (node, timestep) pairs, ranked by true displacement
            magnitude, to compute the direct error over. A handful of nodes
            near the true argmax rather than a single point, so the term isn't
            overly sensitive to exactly which node is the single hardest max
            (mesh noise) while still being tightly localized to the crush zone.
    """
    pos_scale = torch.linalg.norm(pos_std) + 1e-8  # scalar, ~mm per unit normalized
    pred_disp_mm = (pred[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(1, 1, -1)
    target_disp_mm = (target[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(
        1, 1, -1
    )
    pred_mag = torch.linalg.norm(pred_disp_mm, dim=-1).flatten()  # [N*T]
    target_mag = torch.linalg.norm(target_disp_mm, dim=-1).flatten()  # [N*T]

    k = min(top_k, target_mag.numel())
    _, top_idx = torch.topk(target_mag, k)  # no grad through target, fixed locations
    sq_err = torch.square(pred_mag[top_idx] - target_mag[top_idx])
    return torch.mean(sq_err) / (pos_scale**2)


def peak_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_idx: int,
    channel_std: float,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Auxiliary loss term that directly penalizes error at the true peak of a
    single scalar output channel (e.g. effective_plastic_strain), generalizing
    `peak_intrusion_loss`'s top-k-localized approach beyond displacement.

    Identifies the top_k (node, timestep) pairs by GROUND TRUTH value at
    `channel_idx` (fixed locations, no gradient through the ranking — same
    rationale as `peak_intrusion_loss`: a location-agnostic reduction lets the
    optimizer satisfy the term anywhere in the mesh, with no gradient tied to
    where the true peak actually occurs) and computes a direct MSE between
    predicted and true values at exactly those locations, rescaled by
    `channel_std**2` so the term is comparable in magnitude to the base loss
    regardless of the channel's native scale.

    Args:
        pred, target: [N, T, Fo].
        channel_idx: index into the Fo dimension identifying the scalar
            channel to target (e.g. 3 for effective_plastic_strain, given
            channel layout position(0:3) + dynamic_targets in config order).
        channel_std: this channel's std (physical/raw units matching pred and
            target), for rescaling.
        top_k: number of (node, timestep) pairs, ranked by true channel value,
            to compute the direct error over.
    """
    pred_val = pred[:, :, channel_idx].flatten()  # [N*T]
    target_val = target[:, :, channel_idx].flatten()  # [N*T]

    k = min(top_k, target_val.numel())
    _, top_idx = torch.topk(target_val, k)  # no grad through target, fixed locations
    sq_err = torch.square(pred_val[top_idx] - target_val[top_idx])
    return torch.mean(sq_err) / (channel_std**2 + 1e-8)


def combined_crash_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords0: torch.Tensor,
    pos_std: torch.Tensor,
    alpha: float = 2.0,
    peak_loss_weight: float = 0.0,
    peak_loss_top_k: int = 50,
    eps: float = 1e-8,
    channel_std: torch.Tensor | None = None,
    peak_plastic_strain_loss_weight: float = 0.0,
    plastic_strain_channel_idx: int = 3,
) -> torch.Tensor:
    """displacement_weighted_mse + peak_loss_weight * peak_intrusion_loss
    + peak_plastic_strain_loss_weight * peak_field_loss(plastic_strain).

    peak_loss_weight=0.0 and peak_plastic_strain_loss_weight=0.0 exactly
    recover displacement_weighted_mse (backward compatible default).
    """
    base = displacement_weighted_mse(
        pred, target, coords0, pos_std, alpha, eps, channel_std=channel_std
    )
    total = base
    if peak_loss_weight != 0.0:
        peak = peak_intrusion_loss(
            pred, target, coords0, pos_std, top_k=peak_loss_top_k
        )
        total = total + peak_loss_weight * peak
    if peak_plastic_strain_loss_weight != 0.0:
        strain_std = (
            channel_std[plastic_strain_channel_idx].item()
            if channel_std is not None
            else 1.0
        )
        peak_strain = peak_field_loss(
            pred,
            target,
            channel_idx=plastic_strain_channel_idx,
            channel_std=strain_std,
            top_k=peak_loss_top_k,
        )
        total = total + peak_plastic_strain_loss_weight * peak_strain
    return total


def build_muon_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build Muon + AdamW combined optimizer (Muon for 2D params, AdamW for others).

    Muon requires PyTorch >= 2.9. Pass the underlying model (unwrap DDP if needed).
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "Muon optimizer requires PyTorch >= 2.9. "
            "Install a newer PyTorch or use optimizer=adam."
        )
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]
    weight_decay = cfg.training.get("optimizer_weight_decay", 1e-4)
    lr = cfg.training.start_lr
    if muon_params and other_params:
        return CombinedOptimizer(
            [
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                torch.optim.AdamW(
                    other_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1.0e-8,
                ),
            ]
        )
    elif muon_params:
        return torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
    else:
        return torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay)
