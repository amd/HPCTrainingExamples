# Running the MLExamples with Apptainer

README_apptainer.md from `HPCTrainingExamples/MLExamples` in the Training Examples repository

Most examples in this directory only require an installation PyTorch with ROCm support and only few additional packages that can be simply installed via `pip`.
If your system does not provide a PyTorch module, a container is the easiest way to get a consistent, ready-to-use software stack.
This page describes the general workflow how to run the examples in this folder with [Apptainer](https://apptainer.org/), which is the non-root container runtime most HPC centers support.

> [!TIP]
> The `Pytorch/mnist` example contains a complete walkthrough of different ways to run PyTorch in HPC environments, including running it with either Apptainer or Podman: [Pytorch/mnist/README.md](Pytorch/mnist/README.md).

## 1. Get the container image

Most images are published in Docker format and have to be converted to Apptainer's `sif` format once.
Depending on the actual image, the conversion can require significant amounts memory and disk space.
Therefore, perform the conversion on a node that provides sufficient resources for this (e.g. an exclusive allocation) and keep the `.sif` file for future runs.
You can pull and convert the latest image using:

```bash
export SIF=$HOME/rocm-pytorch.sif
apptainer pull $SIF docker://rocm/pytorch:latest
```

Recent `rocm/pytorch` images will work for most exercises. All following commands use the variable `SIF` to refer to the transformed image instead of hardcoding them.
So, make sure to set the variable to the suitable path and you should be able to use all the commands directly.

> [!TIP]
> For training events, a ready-to-use container image is oftentimes prepared beforehand and shared with the participants in a shared location on AAC6, AAC7, or other systems.
> Please follow the specific instructions so you can spend time on the actual exercises instead of waiting for the container conversion to finish.

If the `apptainer pull` commands runs out of space, point Apptainer to a larger directory:
```bash
export APPTAINER_CACHEDIR=/path/with/space/apptainer_cache
export APPTAINER_TMPDIR=/path/with/space/apptainer_tmp
```

## 2. Start the container

On a node with a GPU, e.g. after `salloc --ntasks 1 --gpus=1 --time=01:00:00`, start an interactive shell with
```bash
apptainer shell --cleanenv --rocm $SIF
```

- `--rocm` makes the host's GPU driver and device files available in the container.
- `--cleanenv` stops the host environment (`PYTHONPATH`, `LD_LIBRARY_PATH`, loaded modules, ...) from leaking into the container where it could shadow the image's PyTorch and other pre-installed softare.
  Consequently, the `module` command does not exist in the container and scripts that call `module load` have to be skipped or adapted.
- Your home directory and the current working directory are mounted automatically, so you can access a clone of this repository from the host directly if it is located in the directory you started Apptainer in.

Within the container, frst check that PyTorch sees the GPU:
```bash
python3 -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

## 3. Set up the Python environment of an example

Some of the exercises require additional Python packages beyond the PyTorch installation provided with the (read-only) container image.
There are different approaches how to add additional packages to the read-only container image. Have a look at [the MNIST example](Pytorch/mnist/README.md) to learn about several of them.

If the example does not require additional dependencies with an explicit `requirements.txt` file, you can start the exercise right away and skip the following paragraphs.
In case the examples *does require* additional dependencies, set the environment up like this:

### 3.1. Install additional Python dependencies
A straight-forward approach is to install the additional packages into a virtual environment.
But be aware that a plain `python3 -m venv` inside the container loses access to the pre-installed PyTorch, because the image already uses a virtual environment itself instead of a bare-metal install, so `--system-site-packages` does not point to it.
The `.pth` file below adds the image's site-packages back to the search path of the new environment.
Run this inside the container:
```bash
cd HPCTrainingExamples/MLExamples/<example>
python3 -m venv --system-site-packages --without-pip ./venv-pt
IMAGE_SP=$(python3 -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')
VENV_SP=$(./venv-pt/bin/python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$IMAGE_SP" > "$VENV_SP/_container_torch.pth"
source ./venv-pt/bin/activate
python3 -m pip install -r requirements.txt     # if the example has a requirements.txt
```

> [!WARNING]
> There are a few caveats to be aware of:
>   - The image's Python might not have `ensurepip` installed , so a plain `python3 -m venv` fails with `ensurepip is not available`. You need to add the `--without-pip` flag.
>   - Always install with `python3 -m pip install ...` and not a bare `pip3 install ...`: `pip3` on `PATH` is a script with a hardcoded shebang to the *system* Python, so it silently installs into `$HOME/.local` instead of the venv.
>   - Do not install requirement files that pin `torch`, `triton` or `torchvision` (e.g. `TinyOpenFold/setup/requirements.txt`). This would replace the ROCm-enabled PyTorch of the image with a new wheel. Install only the packages the example needs.

After this one-time setup, only `source ./venv-pt/bin/activate` is needed in later sessions.

### 3.2. Alternative approaches

- **Writable overlay:** keeps changes (venv, cloned repositories) inside a file on the host without modifying the `sif`; see [Pytorch/mnist/README.md](Pytorch/mnist/README.md#extending-the-container-with-a-writable-overlay).
- **`--writable-tmpfs`:** allows temporary writes into the container filesystem that vanish when the container exits.
- **Podman:** see [Pytorch/mnist/README.md](Pytorch/mnist/README.md#podman).

## 4. Non-interactive and batch jobs

Instead of opening an interactive shell with `apptainer shell`, you can also execute commands directly within the container using `apptainer exec`.
You can also pass multiple commands to apptainer for execution by using `bash -s`.
A SLURM script that needs to activate a virtual environment inside the container before running the main Python script could look like this:
```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=apptainer_example.out

SIF=$HOME/rocm-pytorch.sif
cd $HOME/HPCTrainingExamples/MLExamples/<example>

apptainer exec --cleanenv --rocm $SIF bash -s <<'EOS'
source ./venv-pt/bin/activate
python3 <script>.py
EOS
```

## 5. Troubleshooting

- **`torch.cuda.is_available()` is `False`:** check that `--rocm` is passed, that you are on a node with an allocated GPU (`amd-smi`), and that you are in the `render`/`video` groups so `/dev/kfd` and `/dev/dri` are accessible.
- **`ensurepip is not available` when creating the venv:** the image's Python has no `ensurepip`. Add `--without-pip` to the `python3 -m venv` call.
- **Packages installed with `pip3 install` are missing afterwards:** `pip3` on `PATH` points to the system Python, not the venv, and installs land in `$HOME/.local` instead. Use `python3 -m pip install ...` after activating the venv.
- **`ModuleNotFoundError: No module named 'torch'` in the venv:** the `.pth` file is missing or points to the wrong directory. Recreate it as shown in [section 3.1](#3-set-up-the-python-environment-of-an-example).
- **`module: command not found`:** This is expected inside the container, because `--cleanenv` removes the module system. Skip the `module load` lines.
- **`pip install` fails with a network error:** compute nodes often have no direct internet access. Install the packages on a login node (in an interactive container shell) first.
- **`No space left on device`:** redirect the caches as described below and set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR`.

### Caches and writable directories

The image itself is read-only. Many libraries write caches (HuggingFace, MIOpen, Triton, `pip`) into the home directory, which can exhaust quotas.
Redirect them to a directory on a fast, large filesystem, for instance:
```bash
export CACHE=/path/with/space/$USER/cache
mkdir -p $CACHE
export HF_HOME=$CACHE/hf TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton \
       XDG_CACHE_HOME=$CACHE/xdg MIOPEN_USER_DB_PATH=$CACHE/miopen MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen
```
With `--cleanenv`, host environment variables are not passed on. Either `export` them inside the container, or pass them from the host with the `APPTAINERENV_` prefix (e.g. `export APPTAINERENV_HF_HOME=$CACHE/hf`).
The same prefix works for GPU selection (`APPTAINERENV_ROCR_VISIBLE_DEVICES`) and similar variables.
If the cache directory is outside your home, add it with `--bind $CACHE`.
