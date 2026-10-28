"""Optimize two bumper thicknesses using the checkpoint from application 01."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit-mm', type=float, default=350., help='Predicted peak-displacement limit')
    parser.add_argument('--safety-factor', type=float, default=1., help='Explicit multiplier on predictions; not a safety guarantee')
    parser.add_argument('--velocity', type=float, default=-7.)
    parser.add_argument('--wall-position', type=float, default=0.)
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--maxiter', type=int, default=40)
    parser.add_argument('--output-dir', type=Path, help='New results directory relative to the caller; default: timestamped results/ directory')
    args = parser.parse_args()
    import math
    if not all(math.isfinite(v) for v in (args.limit_mm, args.safety_factor, args.velocity, args.wall_position)):
        parser.error('numeric parameters must be finite')
    if args.limit_mm <= 0 or args.safety_factor < 1 or args.grid_size < 2 or args.maxiter < 1:
        parser.error('limit > 0, safety-factor >= 1, grid-size >= 2, maxiter >= 1 required')
    if not -7 <= args.velocity <= -3 or not 0 <= args.wall_position <= 240:
        parser.error('supported load ranges: velocity [-7,-3], wall-position [0,240]')
    root = Path(__file__).resolve().parent
    source = root.parent / '01_training_and_inference'
    output = args.output_dir.resolve() if args.output_dir else root / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Results directory: {output}', flush=True)
    # Existing data/config paths are relative to application 01.
    os.chdir(source)
    sys.path.insert(0, str(source / 'src'))
    import numpy as np
    import torch
    import physicsnemo
    import scipy
    from scipy.optimize import minimize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import bq_torch_patch  # noqa: F401 -- ROCm radius-search backend
    from physicsnemo.utils import load_checkpoint

    started = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = OmegaConf.load(source / 'checkpoint/config.yaml')
    model = instantiate(cfg.model).to(device).eval()
    epoch = load_checkpoint(str(source / 'checkpoint/checkpoints'), models=model, device=device)
    model.requires_grad_(False)  # Freeze weights, preserve derivatives w.r.t. inputs.
    dataset = instantiate(cfg.datapipe, name='optimization', reader=instantiate(cfg.reader), split='test', logger=None)
    base = dataset[0].to(device)
    stats = dict(node={k: v.to(device) for k, v in dataset.node_stats.items()}, edge={},
                 feature={k: v.to(device) for k, v in getattr(dataset, 'feature_stats', {}).items()})
    pos_std = stats['node']['pos_std'].view(1, 1, -1)
    archive = source / 'checkpoint/checkpoints/GeoTransolverOneShot.0.200.mdlus'
    metadata = dict(status='started', checkpoint_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    epoch=epoch, device=str(device), torch=torch.__version__, physicsnemo=physicsnemo.__version__,
                    scipy=scipy.__version__, rocm=torch.version.hip,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    OmegaConf.save(cfg, output / 'checkpoint_config.yaml', resolve=True)
    (output / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Loaded epoch {epoch} on {device}; checkpoint SHA256 {metadata["checkpoint_sha256"]}', flush=True)

    def evaluate(x, gradient=False):
        with torch.set_grad_enabled(gradient):
            design = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=gradient)
            sample = copy.copy(base)
            values = {'velocity_x': design.new_tensor(args.velocity),
                      'crash_box_thick_scale': design[0], 'beam_thick_scale': design[1],
                      'rwall_origin_y': design.new_tensor(args.wall_position)}
            # The model stacks dictionary values: configuration order matters.
            sample.global_features = {key: values[key] for key in cfg.datapipe.global_features}
            pred = model(sample=sample, data_stats=stats)
            displacement = (pred[:, :, :3] - sample.node_features['coords'].unsqueeze(1)) * pos_std
            peak = torch.linalg.vector_norm(displacement, dim=-1).max()
            grad = torch.autograd.grad(peak, design)[0].detach().cpu().numpy().astype(float) if gradient else None
            value = float(peak.detach())
            if not np.isfinite(value) or (gradient and not np.isfinite(grad).all()):
                raise ValueError('Non-finite model prediction or derivative')
            return value, grad

    # A diagnostic, not a pass/fail test: a hard maximum has kinks.
    center = np.ones(2)
    nominal, automatic = evaluate(center, True)
    checks = []
    for h in (0.01, 0.001):
        finite = np.array([(evaluate(center + np.eye(2)[i]*h)[0] - evaluate(center - np.eye(2)[i]*h)[0])/(2*h) for i in range(2)])
        checks.append(dict(step=h, finite_difference=finite.tolist(), absolute_difference=np.abs(finite-automatic).tolist()))
    print(f'Nominal peak: {nominal:.3f} mm; autograd [crash-box, beam]: {automatic} mm/scale', flush=True)
    for row in checks:
        print(f'Central differences h={row["step"]}: {row["finite_difference"]}', flush=True)

    rows = []
    for cb in np.linspace(.7, 1.3, args.grid_size):
        for beam in np.linspace(.7, 1.3, args.grid_size):
            peak, _ = evaluate([cb, beam])
            rows.append(dict(crash_box=float(cb), beam=float(beam), mass_proxy=float((cb+beam)/2),
                             peak_mm=peak, adjusted_peak_mm=args.safety_factor*peak))
    with (output / 'grid.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    feasible = [r for r in rows if r['adjusted_peak_mm'] <= args.limit_mm]
    seed = min(feasible, key=lambda r: r['mass_proxy']) if feasible else min(rows, key=lambda r: r['adjusted_peak_mm'])
    print(f'Grid: {len(feasible)}/{len(rows)} predicted feasible; refining from {seed["crash_box"]:.3f}, {seed["beam"]:.3f}', flush=True)
    cached_x, cached_value = None, None

    def value_grad(x):
        nonlocal cached_x, cached_value
        if cached_x is None or not np.array_equal(x, cached_x):
            cached_value = evaluate(x, True)
            cached_x = x.copy()
        return cached_value

    result = minimize(lambda x: float(np.mean(x)), [seed['crash_box'], seed['beam']],
                      jac=lambda x: np.array([.5, .5]), method='SLSQP', bounds=[(.7, 1.3)]*2,
                      constraints=[dict(type='ineq',
                          fun=lambda x: 1 - args.safety_factor*value_grad(x)[0]/args.limit_mm,
                          jac=lambda x: -args.safety_factor*value_grad(x)[1]/args.limit_mm)],
                      options=dict(maxiter=args.maxiter, ftol=1e-7))
    peak, grad = evaluate(result.x, True)
    violation = max(0., args.safety_factor*peak-args.limit_mm)
    predicted_feasible = violation <= .01 and bool(np.all(result.x >= .7) and np.all(result.x <= 1.3))
    metadata.update(status='evaluated', nominal_peak_mm=nominal, nominal_gradient=automatic.tolist(),
                    gradient_checks=checks, best_feasible_grid= min(feasible, key=lambda r: r['mass_proxy']) if feasible else None,
                    candidate=dict(crash_box=float(result.x[0]), beam=float(result.x[1]), mass_proxy=float(np.mean(result.x)),
                                   peak_mm=peak, adjusted_peak_mm=args.safety_factor*peak, gradient=grad.tolist(),
                                   violation_mm=violation, predicted_feasible=predicted_feasible),
                    solver_success=bool(result.success), solver_message=str(result.message), iterations=int(result.nit),
                    fea_validated=False, total_seconds=time.perf_counter()-started)
    (output / 'result.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    from plot_results import plot_results
    plots = plot_results(output)
    metadata.update(status='complete', plots=plots)
    (output / 'result.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print('Plots: ' + ', '.join(plots), flush=True)
    print(json.dumps({k: metadata[k] for k in ('candidate', 'solver_success', 'solver_message', 'fea_validated')}, indent=2))
    print(f'Results: {output}\nCandidate requires independent FEA validation. No global optimum is established.')


if __name__ == '__main__':
    main()
