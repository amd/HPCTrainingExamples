# Observation provenance

`unseen_shortlist_3/` contains one snapshot from the existing OpenRadioss confirmation
batch `surrogate_opt/confirm_shortlist/run_3`, generated on 2026-09-16 after the
checkpoint's 2026-09-15 training. True crash-box thickness scale is 0.9571428571428571,
beam scale 1.02, velocity -7, wall position 0, and all geometry scales are one.

The raw animation files A001 (initial) and A026 (selected observation) were processed
by `prepare_observation.py`. Their hashes, solver time, training frame label, node
counts, and observation hash are in `observation.json`. `observation.npz` contains
initial coordinates for mesh alignment, structural shell node indices (excluding
auxiliary/contact part 1), and the FEA displacement vectors at the selected frame.
There is no synthetic surrogate output in the observation.

The original bumper model input deck credits Altair Engineering Inc., copyright 2022,
and specifies CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/.
These derived simulation data retain that source attribution and license notice.

`checkpoint_training_manifest.json` records the runs loaded during training from the
checkpoint's original `launch.log`, along with hashes of that log, its Hydra config,
and the checkpoint archive. Training loads the first 220 naturally sorted files from
the merged 225-run set, namely run1..run81 and run201..run339. In particular, run19 and
run201 were in training. The supplied exercise-1 `vtp_holdout` directory means exclusion
from the small toy training loop, not exclusion from the supplied checkpoint's training.

The final five merged cases are not used here: although absent from the loaded training
list, their configurations repeat cases in the original grid. The selected confirmation
case instead has a thickness pair absent from every one of the 225 dataset entries.
It was chosen for a prior surrogate-based design shortlist; it is not a random test sample.

To reproduce extraction given access to the raw confirmation directory:

```bash
python3 prepare_observation.py --vtk-dir /path/to/confirm_shortlist/run_3 --frame 25 --crash-box 0.9571428571428571 --beam 1.02 --velocity -7 --wall-position 0 --output-dir /new/observation/directory
```

Frame k maps to model output k-1. The training converter gives it label k*0.005,
while the source VTK carries a different native solver time; do not interpret the
training label as seconds. The dataset reader takes the first 51 frames without resampling.
