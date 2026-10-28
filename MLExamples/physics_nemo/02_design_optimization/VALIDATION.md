# Validation of the packaged optimizer

Validated on 2026-09-30 with one GPU from the cluster's `MI355x` partition.
The runtime reports `AMD Radeon Graphics`. This run used the existing cluster
environment: PyTorch 2.9.1 / ROCm 7.2.1, PhysicsNeMo 2.1.1, and SciPy 1.18.0.
The ROCm 10 container command in the README has not been rerun for this example.

The successful run used the default arguments with `--output-dir sample_results`.
To reproduce it, select a new output directory:

```bash
python3 -u optimize.py --output-dir results/reproduce
```

The [saved JSON](reported_results/reference_run/result.json) records software versions, device,
configuration arguments, and checkpoint SHA256. The model loaded epoch 200 from
application 01 without training. The grid contains 25 evaluations, 14 predicted feasible.

| Quantity | Measured result |
|---|---:|
| Nominal predicted peak | 344.8167 mm |
| Candidate crash-box scale | 0.700000 |
| Candidate beam scale | 1.100974 |
| Candidate mass proxy | 0.900487 |
| Candidate predicted peak | 350.0000 mm |
| Predicted constraint violation | 0.0000 mm |

SLSQP terminated successfully after three iterations. The mass proxy decreased by
approximately 9.95% relative to nominal. This is neither a measured physical mass
saving nor an FEA-validated design. The saved grid and resolved checkpoint configuration
accompany the JSON. The checkpoint hash matches the supplied application-01 archive.

At nominal thickness, autograd gave `[-43.675991, -88.841858]` mm per thickness scale.
Central differences at step `0.01` gave `[-43.678284, -88.838196]`, with absolute
differences below `0.004 mm/scale`. At step `0.001`, the maximum difference was `0.052`.
This checks the differentiation path at one point, not the model's accuracy against FEA.

An additional GPU run used:

```bash
python3 -u optimize.py --limit-mm 1 --grid-size 2 --maxiter 3 \
  --output-dir results/infeasible-check
```

It reported zero feasible grid points, `solver_success: false` (iteration limit),
`predicted_feasible: false`, and `298.1271 mm` violation. The
[saved result](reported_results/infeasible_check/result.json) confirms that this unsuccessful
search does not receive a feasible label; it does not prove global infeasibility.

The dependency-free `--help` path and repository whitespace checks also passed.
Independent FEA confirmation and testing on other software stacks remain outside
these recorded checks.

## Plot generation

The README figures were generated from the saved `grid.csv` and `result.json`
using `plot_results.py` with Matplotlib 3.11.1. This is postprocessing of the
recorded run, with no additional model inference or changes to its measurements.
Both figures were visually inspected. Plot generation was also exercised on the
saved unsuccessful-search outputs, where every grid point exceeds the limit.
The plotting utility needs no GPU and can be run on results from earlier versions
of the optimization script.
