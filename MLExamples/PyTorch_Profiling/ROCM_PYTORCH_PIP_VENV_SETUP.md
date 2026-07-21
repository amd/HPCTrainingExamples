# Creating the `rocm-pytorch-pip` venv (nightly ROCm + PyTorch, MI300A / gfx942)

This guide walks you through building a Python virtual environment with ROCm,
PyTorch, and the ROCm profiling tools, so you can train and profile a model on
an AMD GPU. Follow the steps in order.

Everything installs from pip into one self-contained venv, so PyTorch and the
profilers use the same ROCm. Each step below is a single command block you can
copy and run.

> This guide targets an AMD MI300A GPU (`gfx942`). If you have a different GPU,
> change `device-gfx942` to your architecture.

---

## 1. Create and activate the venv

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/rocm-pytorch-pip
source ~/venvs/rocm-pytorch-pip/bin/activate
```

## 2. Install ROCm + PyTorch from the nightly multi-arch index

```bash
# Pin the nightly ROCm version once and reuse it everywhere below.
ROCM_VERSION=7.15.0a20260716

pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[profiler,devel,libraries,device-gfx942]==${ROCM_VERSION}" \
    "torch[device-gfx942]" \
    "torchvision[device-gfx942]"
```

The `rocm[...]` extras pull in the pieces this workflow needs:
- `profiler`  — the ROCm profilers: `rocprof-compute`, `rocprofv3`, and
  `rocprof-sys` (bundled `_rocm_profiler`)
- `devel`     — development package (headers/device code, extracted in step 4)
- `libraries` — math libraries (hipBLAS, rocBLAS, ...)
- `device-gfx942` — the GPU-arch kernels for MI300A

## 3. Install `transformers` (required by the training script)

```bash
pip install transformers
```

The training script builds its models with `transformers`, so this package is
required.

> **Note:** Do **not** install `rocprof-compute analyze`'s `requirements.txt`
> into this shared venv. Those packages pin `numpy==1.26.4`, which conflicts
> with the `numpy>=2.0` that PyTorch/transformers need and will break the
> training/profiling runs. The analysis step
> (`rocm-compute-profiler/slurm_single_process_analyze.sh`) provisions its own
> isolated venv for those dependencies and reuses the ROCm install from this
> venv, so this shared venv stays on `numpy>=2.0`.

## 4. Extract development headers and device code

```bash
~/venvs/rocm-pytorch-pip/bin/rocm-sdk init
```

`rocm-sdk init` unpacks the `devel` payload (headers, LLVM device bitcode, etc.)
into `_rocm_sdk_devel/` inside the venv. This provides the device bitcode that
HIP needs to run GPU kernels, and the paths that `setup_rocm.sh` points to in the
next step.

---

## 5. Verify `setup_rocm.sh`

The repo already ships `setup_rocm.sh` in `MLExamples/PyTorch_Profiling/` (the
SLURM scripts source it as `../setup_rocm.sh`). It activates the venv and points
the ROCm environment at the extracted `_rocm_sdk_devel` tree. Verify (and edit
if needed) that its `VENV` matches the venv you built in step 1. The
site-packages path is derived from the active venv's `python3`, so it works
regardless of the Python minor version. Its contents are:

```bash
#!/usr/bin/env bash
# Source this to activate the ROCm venv and set ROCm env vars:
#   source setup_rocm.sh
VENV="$HOME/venvs/rocm-pytorch-pip"
source "$VENV/bin/activate"
DEVEL="$(python3 -c 'import site; print(site.getsitepackages()[0])')/_rocm_sdk_devel"
export ROCM_PATH="$DEVEL"
export HIP_PATH="$DEVEL"
export HIP_DEVICE_LIB_PATH="$DEVEL/lib/llvm/amdgcn/bitcode"
export PATH="$DEVEL/bin:$PATH"
export LD_LIBRARY_PATH="$DEVEL/lib:$DEVEL/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
echo "ROCm venv active: $VENV"
```

## 6. Re-source to pick up the ROCm env vars

If the venv is already active from step 1, deactivate and source the script so
the `ROCM_PATH` / `LD_LIBRARY_PATH` exports take effect (run from
`MLExamples/PyTorch_Profiling/`):

```bash
deactivate
source setup_rocm.sh
```

---

## 7. Verify (on a GPU node)

```bash
source setup_rocm.sh
srun -n1 --gpus=1 python3 -c "import torch; print('torch', torch.__version__); \
x = torch.ones(4, device='cuda:0'); print('device ok:', (x+1).sum().item())"
```

Expected output resembles:

```
ROCm venv active: /.../rocm-pytorch-pip
torch 2.12.0+rocm7.15.0a20260716
device ok: 8.0
```

If you see `device ok:`, the GPU is working and your environment is ready to use.
From now on, just run `source setup_rocm.sh` in any new shell to activate it.
