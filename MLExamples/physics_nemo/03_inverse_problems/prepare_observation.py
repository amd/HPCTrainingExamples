"""Extract one measured displacement snapshot from OpenRadioss animation VTKs.

Preparation only: requires raw FEA output, not the neural network. The workshop
ships a prepared observation so participants do not need the original solver files.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vtk-dir', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=25, help='Frame after t0, 1..50; A001 is t0')
    parser.add_argument('--base-name', default='Bumper_Beam_AP_meshed')
    parser.add_argument('--crash-box', type=float, required=True, help='Ground truth for reporting only')
    parser.add_argument('--beam', type=float, required=True, help='Ground truth for reporting only')
    parser.add_argument('--velocity', type=float, default=-7.)
    parser.add_argument('--wall-position', type=float, default=0.)
    parser.add_argument('--exclude-part', type=int, default=1, help='Exclude auxiliary/contact part 1 from observed shell nodes')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frame <= 50:
        parser.error('frame must be between 1 and 50')
    import numpy as np
    import pyvista as pv

    initial_path = args.vtk_dir / f'{args.base_name}A001.vtk'
    measured_path = args.vtk_dir / f'{args.base_name}A{args.frame+1:03d}.vtk'
    initial, measured = pv.read(initial_path), pv.read(measured_path)
    if not np.array_equal(initial.cells, measured.cells) or initial.n_points != measured.n_points:
        raise ValueError('The two frames must preserve topology and node ordering')
    # Only nodes attached to shell elements are observed; the model still uses
    # its complete original mesh, including other nodes.
    ids = set()
    offset = 0
    cells = initial.cells
    part_ids = initial.cell_data['PART_ID']
    for kind, part in zip(initial.celltypes, part_ids):
        count = int(cells[offset])
        if kind in (5, 9) and part != args.exclude_part:  # structural triangles/quads
            ids.update(cells[offset+1:offset+1+count].tolist())
        offset += count+1
    nodes = np.array(sorted(ids), dtype=np.int64)
    if not len(nodes):
        raise ValueError('No shell nodes in observation mesh')
    coords = np.asarray(initial.points, dtype=np.float64)
    displacement = np.asarray(measured.points, dtype=np.float64) - coords
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / 'observation.npz', coords_mm=coords,
                        node_indices=nodes, displacement_mm=displacement[nodes])
    metadata = dict(frame_after_initial=args.frame, model_output_index=args.frame-1,
                    initial_vtk=initial_path.name, observed_vtk=measured_path.name,
                    initial_vtk_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                    observed_vtk_sha256=hashlib.sha256(measured_path.read_bytes()).hexdigest(),
                    solver_time=float(measured.field_data['TIME'][0]),
                    solver_time_units='native solver time units; no conversion assumed',
                    training_frame_label=args.frame*.005,
                    frame_mapping='Training conversion labels frame k as k*0.005; model uses first 51 frames, without resampling.',
                    velocity_x=args.velocity, rwall_origin_y=args.wall_position,
                    excluded_part_id=args.exclude_part,
                    true_design=dict(crash_box=args.crash_box, beam=args.beam),
                    observed_nodes=int(len(nodes)), total_nodes=int(initial.n_points),
                    source='OpenRadioss FEA; displacement is moving coordinates minus A001 coordinates',
                    observation_sha256=hashlib.sha256((args.output_dir/'observation.npz').read_bytes()).hexdigest())
    (args.output_dir/'observation.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
