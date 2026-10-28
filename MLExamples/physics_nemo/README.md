# AI Surrogates by Example

Guided, hands-on examples of using AI alongside scientific simulations, focused on design
optimization and inverse problems. The bumper workflow starts with training and inference,
then reuses the supplied checkpoint to optimize two independent thickness parameters and
infer them from an unseen FEA displacement snapshot.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/physicsnemo/main/docs/img/crash/bumper_beam.gif" alt="PhysicsNeMo bumper-beam crash animation" width="80%" />
</p>

Animation from [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash).
This illustrates the bumper example; it is not an output from the workshop measurements.

| Example | Scope | Current stage |
|---|---|---|
| [01_training_and_inference](01_training_and_inference) | Toy training, real model size, inference timing, and error against two simulations | Available |
| [02_design_optimization](02_design_optimization) | Optimize independent crash-box and beam thicknesses using the supplied checkpoint and automatic derivatives | Available |
| [03_inverse_problems](03_inverse_problems) | Infer crash-box and beam thicknesses from one unseen FEA snapshot using automatic derivatives | Available |

## Shared environment

Use the prebuilt ROCm 10 PyTorch image. If you do not already have it, pull it
to a directory outside the checkout on the host:

```bash
mkdir -p "$HOME/containers"
apptainer pull "$HOME/containers/workshop_rocm10_pytorch.sif" docker://rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0
```

From `MLExamples/physics_nemo` on an allocated GPU node, enter the container:

```bash
apptainer shell --rocm --bind "$(pwd):/examples" --pwd /examples "$HOME/containers/workshop_rocm10_pytorch.sif"
```

Substitute your existing image path if it differs. Your home directory must be mounted
(Apptainer normally mounts it automatically). Clear any `PYTHONPATH` left from an older
installation before setting up or running the exercises: `unset PYTHONPATH`.

For the prebuilt ROCm 10 PyTorch container with `/opt/venv/bin/python3`, run
the shared installer from this directory inside the container:

```bash
bash setup_env_rocm10_pytorch_venv.sh
```

It creates `$HOME/venvs/physicsnemo-rocm10`, reuses the container's PyTorch, and
installs the workshop dependencies including SciPy. Pass a different new directory
as the first argument to change the destination. This environment supports all three
exercises. From this directory inside the container,
set up single-process execution:

```bash
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
```

Run exercise 1 (Adam for the MI300A toy training loop):

```bash
FORCE_ADAM_MI300A=1 "$HOME/venvs/physicsnemo-rocm10/bin/python3" -u 01_training_and_inference/workshop_crash_surrogate.py --epochs 30
```

Or run exercise 2:

```bash
"$HOME/venvs/physicsnemo-rocm10/bin/python3" -u 02_design_optimization/optimize.py --limit-mm 350 --output-dir 02_design_optimization/results/first_run
```

Run exercise 3 to infer thicknesses from a measured displacement snapshot:

```bash
"$HOME/venvs/physicsnemo-rocm10/bin/python3" -u 03_inverse_problems/infer.py --output-dir 03_inverse_problems/results/first_run
```

These commands need neither activation nor a `PYTHONPATH` setting. The environment
persists in your home directory and must be used inside the same
container image. Setup is required only once; future container sessions can use the
environment's Python directly. The installer refuses to overwrite an existing directory.
See the [optimization instructions](02_design_optimization/README.md#run) and
[training instructions](01_training_and_inference/README.md#setup-and-run) for exercise options.

## Workshop progression

Start with **training and inference**: use PhysicsNeMo to predict bumper deformation, plastic
strain, and stress from a mesh and design and impact parameters. Train a small toy model, then
load the separate supplied checkpoint and measure its size, inference speed, and error against
two simulations. Data preparation, model accuracy, and inference speed support the applications.

Continue with **design optimization**: reload the same checkpoint to search independent crash-box
and beam thicknesses. Minimize a mass proxy subject to a predicted peak-displacement limit using
a grid scan followed by SLSQP with automatic derivatives. The example explains the mathematics,
checks gradients against finite differences, and reports predicted feasibility. Proposed designs
still require independent FEA validation.

**Inverse problems** recover the two thicknesses from an FEA displacement snapshot at one
frame. The observed configuration was absent from checkpoint training. The example fits
the displacement vectors with autograd, checks recovery against the true parameters,
and saves field comparisons, a fit-error landscape, and convergence plots.

## Results

The training and inference example includes a supplied checkpoint and
[verification notes](01_training_and_inference/TESTING.md). The optimization example includes
[GPU validation notes](02_design_optimization/VALIDATION.md) and
[saved results](02_design_optimization/reported_results/reference_run/result.json) from that checkpoint,
including the configuration, gradient checks, and optimizer outcome. These results describe
surrogate predictions and a mass proxy, not FEA-confirmed safety or measured physical mass savings.

Each script saves measurements, configuration, and figures under its exercise's `results/`
directory. Selected reference runs are included under `reported_results/` and illustrated in
the exercise READMEs. The inverse exercise includes its measurements and training-exclusion
evidence in [its README](03_inverse_problems/README.md).
