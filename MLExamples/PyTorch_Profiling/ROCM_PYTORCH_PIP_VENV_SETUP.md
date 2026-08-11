# Creating the `rocm-pytorch-pip` venv (nightly ROCm + PyTorch)

This guide builds a Python virtual environment holding ROCm, PyTorch, and the
ROCm profiling tools, so you can train and profile a model on an AMD GPU.
Everything installs from pip into one self-contained venv, so PyTorch and the
profilers use the same ROCm.

> `install_rocm_pytorch.sh` performs every step below and pre-stages the test
> jobs. Follow the steps by hand only to adapt the build or debug a failure; see
> [`README_ROCM_NIGHTLY_TESTING.md`](./README_ROCM_NIGHTLY_TESTING.md) for the
> scripted path.

The commands use the settings from `local.env`, so create one first and let
`env.sh` load it. Nothing below is specific to a cluster or a GPU:

```bash
cp local.env.example local.env
$EDITOR local.env          # VENV_BASE, GPU_ARCH, ROCM_VERSION at minimum
source env.sh
```

---

## 1. Create and activate the venv

`VENV_BASE` holds the `venvs` folder. It must be visible from **both** the login
node that builds the venv and the compute nodes that run the jobs, so use the
fastest *shared* filesystem available (a parallel filesystem where there is one).
Node-local paths such as `/tmp` or `/dev/shm` cannot work here, however fast they
are: the compute node would not see what the login node wrote.

```bash
mkdir -p "${VENV_BASE}/venvs"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
```

## 2. Install ROCm + PyTorch from the nightly multi-arch index

```bash
pip install --index-url "${ROCM_INDEX_URL}" \
    "rocm[profiler,devel,libraries,device-${GPU_ARCH}]==${ROCM_VERSION}" \
    "torch[device-${GPU_ARCH}]" \
    "torchvision[device-${GPU_ARCH}]"
```

The `rocm[...]` extras pull in the pieces this workflow needs:
- `profiler`  — the ROCm profilers: `rocprof-compute`, `rocprofv3`, and
  `rocprof-sys` (bundled `_rocm_profiler`)
- `devel`     — development package (headers/device code, extracted in step 4)
- `libraries` — math libraries (hipBLAS, rocBLAS, ...)
- `device-${GPU_ARCH}` — the GPU-arch kernels (`gfx942` for MI300A, `gfx90a` for
  MI250X); see the index for the architectures a given nightly ships

To validate a different nightly, change `ROCM_VERSION` in `local.env` and
rebuild. Available versions are listed at the index URL itself,
<https://rocm.nightlies.amd.com/whl-multi-arch/>.

## 3. Install `transformers` (required by the training script)

```bash
pip install transformers
```

The training script builds its models with `transformers`, so this package is
required.

## 4. Extract development headers and device code

```bash
"${VENV}/bin/rocm-sdk" init
```

`rocm-sdk init` unpacks the `devel` payload (headers, LLVM device bitcode) into
`_rocm_sdk_devel/` inside the venv — the device bitcode HIP needs to run kernels,
and the paths `setup_rocm.sh` points at next.

---

## 5. Activate with `setup_rocm.sh`

The repo ships `setup_rocm.sh` in `MLExamples/PyTorch_Profiling/` (the SLURM
scripts source it as `../setup_rocm.sh`). It reads `local.env` through `env.sh`,
activates the venv, points the ROCm environment at the extracted
`_rocm_sdk_devel` tree, and keeps MIOpen's databases on node-local storage.
There is nothing in it to edit:

```bash
deactivate                 # if the venv is still active from step 1
source setup_rocm.sh
```

Where `install_rocm_pytorch.sh` has generated an Lmod modulefile under
`${VENV_BASE}/modulefiles`, `setup_rocm.sh` loads that module instead of
activating the venv directly. Either way the resulting environment is the same.

## 6. Verify (on a GPU node)

```bash
source setup_rocm.sh
srun -n1 --gpus=1 python3 -c "import torch; print('torch', torch.__version__); \
x = torch.ones(4, device='cuda:0'); print('device ok:', (x+1).sum().item())"
```

Expected output resembles:

```
setup_rocm.sh: ROCm venv active: /.../rocm-pytorch-pip
torch 2.12.0+rocm7.15.0a20260721
device ok: 8.0
```

If you see `device ok:`, the GPU is working and your environment is ready to use.
From now on, just run `source setup_rocm.sh` in any new shell to activate it.
