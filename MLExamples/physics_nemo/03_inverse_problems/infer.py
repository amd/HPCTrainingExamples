"""Infer two bumper thicknesses from one FEA displacement snapshot."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-dir', type=Path, default=root/'data/unseen_shortlist_3')
    parser.add_argument('--output-dir', type=Path, help='New directory relative to the caller; default: results/<timestamp>')
    parser.add_argument('--grid-size', type=int, default=7)
    parser.add_argument('--maxiter', type=int, default=50)
    parser.add_argument('--noise-mm', type=float, default=0., help='Optional independent Gaussian noise std per displacement component')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    import math
    if args.grid_size < 2 or args.maxiter < 1 or not math.isfinite(args.noise_mm) or args.noise_mm < 0 or args.seed < 0:
        parser.error('grid-size >= 2, maxiter >= 1, finite noise-mm >= 0, seed >= 0 required')
    observation_dir = args.observation_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else root/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Results: {output}', flush=True)
    source = root.parent/'01_training_and_inference'
    os.chdir(source)
    os.environ['WORKSHOP_VTP_OUTPUT_DIR'] = str(output/'reader_vtp')
    sys.path.insert(0, str(source/'src'))
    import numpy as np
    import torch
    import scipy
    import physicsnemo
    from scipy.optimize import minimize
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import bq_torch_patch  # noqa: F401 -- use the workshop's ROCm distance backend
    from physicsnemo.utils import load_checkpoint

    start = time.perf_counter()
    obs = json.loads((observation_dir/'observation.json').read_text())
    observation_path = observation_dir/'observation.npz'
    if hashlib.sha256(observation_path.read_bytes()).hexdigest() != obs['observation_sha256']:
        raise ValueError('Observation hash does not match its provenance')
    with np.load(observation_path, allow_pickle=False) as data:
        coords_mm = data['coords_mm'].copy()
        nodes = data['node_indices'].copy()
        clean = data['displacement_mm'].copy()
    if clean.shape != (len(nodes), 3) or coords_mm.ndim != 2 or coords_mm.shape[1] != 3:
        raise ValueError('Expected Nx3 coordinates and one displacement vector per observed node')
    if not np.isfinite(clean).all() or not np.isfinite(coords_mm).all() or len(nodes) == 0:
        raise ValueError('Observations must be finite and nonempty')
    if nodes.min() < 0 or nodes.max() >= len(coords_mm) or len(np.unique(nodes)) != len(nodes):
        raise ValueError('Invalid observed node indices')
    observed = clean + np.random.default_rng(args.seed).normal(0., args.noise_mm, clean.shape)
    scale2 = float(np.mean(np.sum(observed**2, axis=1)))
    if scale2 <= 1e-12:
        raise ValueError('A zero-displacement snapshot cannot identify thickness')
    cfg = OmegaConf.load(source/'checkpoint/config.yaml')
    frame = obs['model_output_index']
    if not isinstance(frame, int) or not 0 <= frame < cfg.training.num_time_steps-1:
        raise ValueError('Observation frame must correspond to a predicted output frame')
    archive = source/'checkpoint/checkpoints/GeoTransolverOneShot.0.200.mdlus'
    checkpoint_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = json.loads((root/'data/checkpoint_training_manifest.json').read_text())
    bundled = observation_dir == (root/'data/unseen_shortlist_3').resolve()
    if bundled and checkpoint_hash != manifest['checkpoint_sha256']:
        raise ValueError('Bundled holdout evidence applies only to the documented checkpoint')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = instantiate(cfg.model).to(device).eval()
    epoch = load_checkpoint(str(source/'checkpoint/checkpoints'), models=model, device=device)
    model.requires_grad_(False)
    dataset = instantiate(cfg.datapipe, name='inverse_mesh', reader=instantiate(cfg.reader), split='test', logger=None)
    base = dataset[0].to(device)
    stats = dict(node={k:v.to(device) for k,v in dataset.node_stats.items()}, edge={},
                 feature={k:v.to(device) for k,v in getattr(dataset, 'feature_stats', {}).items()})
    pos_std = stats['node']['pos_std'].view(1, 3)
    pos_mean = stats['node']['pos_mean'].view(1, 3)
    base_coords = (base.node_features['coords']*pos_std+pos_mean).detach().cpu().numpy()
    if base_coords.shape != coords_mm.shape or not np.allclose(base_coords, coords_mm, atol=1e-3, rtol=0):
        raise ValueError('Observation geometry/node order does not match the checkpoint mesh')
    # The base sample supplies topology and output shape, never the target response
    # or thickness labels used for fitting. All global features are replaced below.
    base.node_target = torch.zeros_like(base.node_target)
    node_ids = torch.tensor(nodes, dtype=torch.long, device=device)
    target = torch.tensor(observed, dtype=torch.float32, device=device)
    metadata = dict(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), torch=torch.__version__, scipy=scipy.__version__,
                    physicsnemo=physicsnemo.__version__, rocm=torch.version.hip,
                    device=str(device), device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    checkpoint_sha256=checkpoint_hash, checkpoint_epoch=epoch,
                    observation=obs, observed_nodes=len(nodes), noise_mm=args.noise_mm, seed=args.seed,
                    grid_size=args.grid_size, maxiter=args.maxiter,
                    holdout_verified=bundled, holdout_evidence='data/checkpoint_training_manifest.json' if bundled else 'Unverified custom observation',
                    observation_rms_mm=math.sqrt(scale2))
    OmegaConf.save(cfg, output/'checkpoint_config.yaml', resolve=True)
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Loaded epoch {epoch}; fitting {len(nodes)} node vectors at output index {frame}', flush=True)

    def forward(x):
        sample = copy.copy(base)
        values = {'velocity_x': x.new_tensor(obs['velocity_x']), 'crash_box_thick_scale':x[0],
                  'beam_thick_scale':x[1], 'rwall_origin_y':x.new_tensor(obs['rwall_origin_y'])}
        sample.global_features = {key:values[key] for key in cfg.datapipe.global_features}
        prediction = model(sample=sample, data_stats=stats)
        return ((prediction[:,frame,:3]-sample.node_features['coords'])*pos_std)[node_ids]

    def evaluate(x, gradient=False):
        with torch.set_grad_enabled(gradient):
            design = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=gradient)
            predicted = forward(design)
            loss = ((predicted-target)**2).sum(dim=1).mean()/scale2
            grad = torch.autograd.grad(loss, design)[0].detach().cpu().numpy().astype(float) if gradient else None
            value = float(loss.detach())
        if not math.isfinite(value) or (gradient and not np.isfinite(grad).all()):
            raise ValueError('Non-finite objective or gradient')
        return value, grad

    center = np.ones(2)
    initial_loss, automatic = evaluate(center, True)
    checks = []
    for h in (.01, .001):
        fd = np.array([(evaluate(center+np.eye(2)[i]*h)[0]-evaluate(center-np.eye(2)[i]*h)[0])/(2*h) for i in range(2)])
        checks.append(dict(step=h, finite_difference=fd.tolist(), absolute_difference=np.abs(fd-automatic).tolist()))
    print(f'Nominal vector RMSE: {math.sqrt(initial_loss*scale2):.4f} mm; autograd: {automatic}', flush=True)
    print('Finite-difference checks: '+json.dumps(checks), flush=True)
    grid = []
    for cb in np.linspace(.7,1.3,args.grid_size):
        for beam in np.linspace(.7,1.3,args.grid_size):
            loss,_ = evaluate([cb,beam])
            grid.append(dict(crash_box=float(cb), beam=float(beam), normalized_mse=loss, vector_rmse_mm=math.sqrt(loss*scale2)))
    write_csv(output/'grid.csv', grid)
    best_grid = min(grid, key=lambda row:row['normalized_mse'])
    seeds = []
    for seed in [[best_grid['crash_box'],best_grid['beam']], [1.,1.], [.75,1.25], [1.25,.75], [.75,.75]]:
        if seed not in seeds:
            seeds.append(seed)
        if len(seeds)==4:
            break
    history, fits = [], []
    for start_id, seed in enumerate(seeds):
        cache_x, cache_result = None, None
        def objective(x):
            nonlocal cache_x, cache_result
            if cache_x is None or not np.array_equal(cache_x,x):
                cache_result = evaluate(x, True)
                cache_x = x.copy()
            return cache_result
        iteration = 0
        def record(x):
            nonlocal iteration
            value,_ = objective(np.asarray(x))
            history.append(dict(start=start_id, iteration=iteration, crash_box=float(x[0]), beam=float(x[1]),
                                normalized_mse=value, vector_rmse_mm=math.sqrt(value*scale2)))
            iteration += 1
        record(seed)
        solution = minimize(objective, seed, jac=True, method='L-BFGS-B', bounds=[(.7,1.3)]*2,
                            callback=record, options=dict(maxiter=args.maxiter, ftol=1e-8, gtol=1e-6, maxls=30))
        final_loss,_ = evaluate(solution.x)
        fits.append(dict(start=start_id, initial_crash_box=seed[0], initial_beam=seed[1],
                         crash_box=float(solution.x[0]), beam=float(solution.x[1]), normalized_mse=final_loss,
                         vector_rmse_mm=math.sqrt(final_loss*scale2), success=bool(solution.success),
                         iterations=int(solution.nit), message=str(solution.message)))
        print(f'Start {start_id}: thicknesses {solution.x}; vector RMSE {math.sqrt(final_loss*scale2):.4f} mm; success={solution.success}', flush=True)
    write_csv(output/'optimization_history.csv', history)
    write_csv(output/'multistart.csv', fits)
    best = min(fits, key=lambda row:row['normalized_mse'])
    # Prefer a converged search when the difference is below float32 objective
    # resolution. Preserve every status and value in multistart.csv.
    tied = [row for row in fits if row['success'] and row['normalized_mse'] <= best['normalized_mse']+1e-8]
    if tied:
        best = min(tied, key=lambda row:row['normalized_mse'])
    selected_source = 'optimizer'
    if best_grid['normalized_mse'] < best['normalized_mse']:
        best = best_grid
        selected_source = 'grid_fallback'
    estimate = [best['crash_box'],best['beam']]
    # Reveal the original thicknesses only AFTER all fitting is finished.
    truth = [obs['true_design']['crash_box'],obs['true_design']['beam']]
    with torch.no_grad():
        fitted = forward(torch.tensor(estimate, dtype=torch.float32, device=device)).cpu().numpy()
        nominal = forward(torch.ones(2, device=device)).cpu().numpy()
        at_truth = forward(torch.tensor(truth, dtype=torch.float32, device=device)).cpu().numpy()
    np.savez_compressed(output/'snapshot.npz', coords_mm=coords_mm[nodes], node_indices=nodes,
                        observed_mm=observed, clean_fea_mm=clean, fitted_mm=fitted,
                        nominal_mm=nominal, surrogate_at_true_design_mm=at_truth)
    metadata.update(status='evaluated', nominal_gradient=automatic.tolist(), gradient_checks=checks,
                    selection=selected_source, selected_start=best.get('start'),
                    solver_success=best.get('success',False), solver_message=best.get('message','Best grid point retained'),
                    inferred_design=dict(crash_box=estimate[0],beam=estimate[1]),
                    absolute_parameter_error=dict(crash_box=abs(estimate[0]-truth[0]), beam=abs(estimate[1]-truth[1])),
                    fitted_vector_rmse_mm=float(np.sqrt(np.mean(np.sum((fitted-observed)**2,axis=1)))),
                    nominal_vector_rmse_mm=math.sqrt(initial_loss*scale2),
                    true_design_surrogate_vector_rmse_mm=float(np.sqrt(np.mean(np.sum((at_truth-observed)**2,axis=1)))),
                    relative_field_error=float(np.linalg.norm(fitted-observed)/np.linalg.norm(observed)),
                    normalized_mse=best['normalized_mse'], fea_validated_inferred_design=False,
                    fitting_seconds=time.perf_counter()-start)
    (output/'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    from plot_results import plot_results
    metadata['plots'] = plot_results(output)
    metadata['status'] = 'complete'
    (output/'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:metadata[k] for k in ['inferred_design','absolute_parameter_error','fitted_vector_rmse_mm',
                    'true_design_surrogate_vector_rmse_mm','relative_field_error','solver_success']},indent=2))
    print(f'Results and figures: {output}\nRecovered a surrogate fit; no uniqueness or independent FEA validation is claimed.')


if __name__ == '__main__':
    main()
