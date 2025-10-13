
# Overview of ML and AI on AMD GPUs

README_AAC6.md in HPCTrainingExamples/MLExamples

## PyTorch wheel install

AAC6 has ROCm 6.4.1. To download PyTorch

```
pip3 install --target temp/pip-installs --pre torch==2.8.0+rocm6.4 --index-url https://download.pytorch.org/whl
```

You will see the following:

```
Looking in indexes: https://download.pytorch.org/whl
Collecting torch==2.8.0+rocm6.4
  Downloading https://download.pytorch.org/whl/rocm6.4/torch-2.8.0%2Brocm6.4-cp310-cp310-manylinux_2_28_x86_64.whl (3607.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.6/3.6 GB 528.2 kB/s eta 0:00:00
....
```

We can use the python packages that we just installed by setting the PYTHONPATH to the location where we installed PyTorch

```
module load rocm
export PYTHONPATH=temp/pip-installs
srun -n1 --gpus=4 \
	python3 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
```

Cleaning up after this, remove the temp directory

```
rm -rf temp
```

## PyTorch module

AAC6 has a module with Pytorch and friends pre-installed on the system. To use that version, unset
the PYTHONPATH and load the module.

```
unset PYTHONPATH
module load rocm pytorch
srun -n1 --gpus==4 \
	python3 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
```

To clean up after, all we have to do is unload the module

```
module unload pytorch
```

## PyTorch virtual environment

Setting up a virtual environment to avoid conflicts and leftover packages. Once we have the virtual
environment setup, we do not need to specify install location – the
environment is doing it for you.

```
python3 -m venv pytorch_install
source pytorch_install/bin/activate
python3 -m pip install --pre torch==2.8.0+rocm6.4 --index-url https://download.pytorch.org/whl
srun -n1 --gpus==4 \
	python3 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
```

Cleaning up afterwards

```
deactivate
rm -rf pytorch_install
```

## PyTorch with a conda environment

The advantage of using a conda environment is especially apparent when you want to use a different
version of python. To use 

```
module load miniconda
conda create -y -n pytorch python=3.10
conda activate pytorch
```

We have to reinstall pytorch because the conda version does not have the GPUs enabled

```
python3 -m pip install --pre torch==2.8.0+rocm6.4 --index-url https://download.pytorch.org/whl
srun -n1 --gpus=4 --nodelist ppac-pl1-s24-26 python3 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
```

Be sure and clean up afterwards.

```
conda deactivate
module unload miniconda
```

## TensorFlow wheel install in a virtual environment

```
python3 -m venv tensorflow-install
source tensorflow-install
python3 -m pip install tensorflow-rocm==2.14.0.600
srun -n1 --gpus=4 python3 python3 -c 'from tensorflow.python.client import device_lib ; device_lib.list_local_devices()'
```

If you get an error, downgrade numpy as follows

```
python3 -m pip install 'numpy<2'
srun -n1 --gpus=4 python3 -c 'from tensorflow.python.client import device_lib ; device_lib.list_local_devices()'
```

Output should be:

```
2025-10-12 14:59:16.842549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /device:GPU:0 with 94875 MB memory:  -> device: 0, name: AMD Instinct MI300A, pci bus id: 0000:01:00.0
2025-10-12 14:59:18.466081: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /device:GPU:1 with 94875 MB memory:  -> device: 1, name: AMD Instinct MI300A, pci bus id: 0001:01:00.0
2025-10-12 14:59:18.812222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /device:GPU:2 with 94875 MB memory:  -> device: 2, name: AMD Instinct MI300A, pci bus id: 0002:01:00.0
2025-10-12 14:59:19.148806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /device:GPU:3 with 94875 MB memory:  -> device: 3, name: AMD Instinct MI300A, pci bus id: 0003:01:00.0
```

And cleaning up afterwards

```
deactivate
rm -rf tensorflow-install
```


## Tensorflow with module

```
module load tensorflow
srun -n1 --gpus=4 python3 -c 'from tensorflow.python.client import device_lib ; device_lib.list_local_devices()'
```

Cleaning up

```
module unload
```

## JAX wheel install in a virtual environment

```
python3 -m venv jax-install
source jax-install/bin/activate
python3 -m pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jaxlib-0.5.0-cp310-cp310-manylinux_2_28_x86_64.whl
python3 -m pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jax_rocm60_plugin-0.5.0-cp310-cp310-manylinux_2_28_x86_64.whl
python3 -m pip install https://github.com/ROCm/jax/archive/refs/tags/rocm-jax-v0.5.0.tar.gz
srun -n1 --gpus=4 --nodelist ppac-pl1-s24-26 python -c 'import jax ; print("I have this many GPUs:", jax.local_device_count())'
```

may be getting an error currently

## JAX with a module

```
module load rocm jax
srun -n1 --gpus=4 --nodelist ppac-pl1-s24-26 python -c 'import jax ; print("I have this many GPUs:", jax.local_device_count())'
```

Output

```
I have this many GPUs: 4
```

