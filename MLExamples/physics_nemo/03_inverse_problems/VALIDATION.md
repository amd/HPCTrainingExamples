# Validation

The published run in `reported_results/first_run/` was executed on 2026-09-30 using
one GPU allocated from the `MI355x` partition. It used the existing cluster environment;
full versions and device identification are in `run.json`. Testing of this exercise
in the shared ROCm 10 image remains separate from these measurements.

## Provenance and independence

- The original checkpoint training log lists exactly 220 unique simulation IDs before
  validation starts. The configuration requests 220 training samples. The reader loads
  naturally sorted files and stops at that count. The loaded list includes run19/run201.
- The bundled observation's thickness pair is absent from all 225 merged configuration
  records, and its FEA batch ran after checkpoint training. The source pair is recorded
  independently from the fitting objective and is not used to select starts.
- Snapshot extraction checks identical connectivity/node ordering between raw initial
  and observed VTK frames. The source hashes and frame indices are recorded. Auxiliary
  part 1 is excluded from observed nodes; the whole mesh is retained for model evaluation.
- The fitter checks the observation hash, the checkpoint hash for the bundled case,
  and initial-coordinate alignment with the checkpoint mesh to 0.001 mm.
- The selected frame is A026, model output index 24; there is no resampling or interpolation
  in time. The historical training frame label and native solver time are both retained.

## Numerical checks

At nominal thickness, autograd returned approximately `[-0.00012078, 0.00045897]`
for the dimensionless loss gradient. Central differences at step 0.01 were
`[-0.00012114, 0.00046065]`, within about 0.4% per component. The 0.001-step
comparison has more float32 cancellation; raw values remain in `run.json`.

All four starts converged successfully in the published run, with nearly identical
parameters and vector RMSE. They recovered approximately `(1.0206, 0.9924)` compared
with the true `(0.9571, 1.0200)`. Fitted vector RMSE is 4.1784 mm; relative field error
is 3.4003%. These numbers were independently recomputed from the saved snapshot arrays.
The surrogate at the true parameters has RMSE 4.2413 mm, demonstrating model discrepancy.
Multiple converged starts do not establish a unique physical inverse solution.

The three figures were rendered from the saved artifacts and visually inspected.
The dependency-free help paths, Python syntax, notebook JSON after the holdout-label
correction, and local documentation/figure links were checked. No new neural-network
training was performed, and no FEA run at the inferred design is claimed.

The optional observation-noise setting is reproducible from the seed and saves both
clean and noisy observations. It is an extension for participants; the reference
measurement uses zero added noise.
