"""Force PhysicsNeMo radius_search onto the pure-torch backend, and replace
torch.cdist globally with a manual pairwise-distance implementation.

On AMD/ROCm, `warp-lang` imports but has no HIP backend, so it registers as
`available=True` yet fails at runtime (IndexError: cuda_devices[0]) when
GeoTransolver's BQWarp runs its ball query. Warp is rank 0, so default dispatch
picks it over the shipped pure-torch backend (rank 1).

Importing this module marks the warp backend unavailable, so dispatch selects
the torch backend (validated to match warp's padding convention and to run on
the MI300X). Local features stay ON — full-accuracy GeoTransolver/GeoFLARE.

The torch backend's `_torch_impl.py` calls
`torch.cdist(points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")`
to compute pairwise distances for the ball query. In the environments we
tested (torch 2.9.1-2.13.0 paired with ROCm 7.2.4, 7.14.1, and 10.0.0 on
MI300A), this specific call did not produce usable results:

  - ROCm 10.0.0 / 7.14.1: the call raises `hipErrorInvalidConfiguration` at
    every input size we tried (4096-16384), including the real ~13.9k-node
    mesh used by this workshop.
  - ROCm 7.2.4: the call does not raise, but returns distances that don't
    match a CPU float64 reference computation for a large fraction of
    pairs (frequently exactly 0.0). The identical call on CPU, and the
    identical call on GPU with the default (unset) `compute_mode`, both
    match the reference -- only this specific compute_mode's GPU kernel
    path disagreed in our testing. See TESTING.md for the full
    investigation and numbers.

We also benchmarked forcing `compute_mode="use_mm_for_euclid_dist_if_necessary"`
(the matmul/Gram-trick alternative) as a possible fix: it was ~35x slower on
this mesh AND produced values inconsistent with the CPU reference for pairs
within the actual ball-query radius (consistent with catastrophic
cancellation in `|x|^2+|y|^2-2xy`, which is presumably why PhysicsNeMo's own
code comment already calls the unset-compute_mode default "numerically
unstable" for this use case) -- so that flag isn't a viable substitute here.

The fix below replaces torch.cdist globally (for this process) with a manual
broadcast-based distance (elementwise sub/pow/sum/sqrt). This is a different
code path than any cdist compute_mode, so it isn't affected by whichever
kernel path was disagreeing with the reference above. Checked against the
same CPU float64 reference and against a full run of this workshop:

  - Matches the CPU float64 reference exactly on the cases we checked.
  - No exception at any tested size (4096-16384) on ROCm10.
  - ~3% forward-pass overhead relative to the native `donot_use_mm` kernel,
    measured on ROCm 7.2.4 (6.43ms vs 6.22ms at N=13882) -- and this is only
    called a handful of times per forward pass, not on the model's dominant
    compute path.
  - End-to-end: BQWarp on the real ~13.9k-node mesh runs on ROCm10 and
    returns neighbors correctly within the query radius; the existing
    pretrained checkpoint (no retraining) evaluates with accuracy matching
    the baseline previously recorded in TESTING.md.

Import once, before any model is built:  `import bq_torch_patch`
"""
from dataclasses import replace

import torch

from physicsnemo.nn.functional.neighbors.radius_search.radius_search import (
    RadiusSearch,
)


def _safe_cdist(
    x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0, compute_mode: str | None = None
) -> torch.Tensor:
    """Global replacement for torch.cdist (see module docstring for why).

    p=2.0 (Euclidean -- the only value physicsnemo's radius_search passes,
    and by far the most common in practice) uses the manual broadcast
    sub/pow/sum/sqrt checked against a CPU float64 reference. `p=inf` and
    general finite `p` use the same broadcast-diff approach with the
    general Lp-norm formula, so this is a safe drop-in for any caller, not
    just radius_search. `compute_mode` is accepted for signature
    compatibility but ignored -- there's only one code path here.
    """
    diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
    if p == 2.0:
        return diff.pow(2).sum(-1).clamp_min(0).sqrt()
    if p == float("inf"):
        return diff.amax(-1)
    return diff.pow(p).sum(-1).clamp_min(0).pow(1.0 / p)


def apply() -> str:
    impls = RadiusSearch._get_impls()
    if "warp" in impls and impls["warp"].available:
        impls["warp"] = replace(impls["warp"], available=False)
    selected = sorted(
        (i for i in impls.values() if i.available), key=lambda i: i.rank
    )[0].name
    return selected


_selected = apply()
print(f"[bq_torch_patch] radius_search backend forced -> '{_selected}'")

# Patched globally (not scoped to radius_search's call sites) since the
# behavior described above is in torch.cdist itself, not in how
# physicsnemo calls it -- other cdist call sites elsewhere in physicsnemo
# (lagrangian_dataset.py, diffusion preconditioners, figconvnet,
# mesh_lsq_gradient.py) would see the same behavior if ever exercised with
# this compute_mode on this hardware, even though this workshop doesn't
# currently touch those paths.
torch.cdist = _safe_cdist
print("[bq_torch_patch] torch.cdist patched globally -> _safe_cdist (see module docstring)")
