
# AI and ML exercises

README_OnInstinctNode.md in HPCTrainingExamples/MLExamples

Last revision of this README: **April 14th 2025**.

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

Throughout these exercises we'll be leveraging the existing ROCm installation. We can use the existing module to set the environment for it:

```
module purge
module load rocm
```

## Setting the virtual environments
These exercises include use cases for PyTorch and TensorFlow using Horovod. Let's prepare the environments to install these frameworks.

We'll be leveraging the system Python installation, so we'll be creating virtual environments to add the Python packages we need:

Let's create a virtual environment for PyTorch:
```
python3 -m venv --system-site-packages $HOME/venv-pt
```

And one for Tensorflow:
```
python3 -m venv --system-site-packages $HOME/venv-tf
```

## Installing the frameworks
Let's install a PyTorch and Tensorflow suitable for the ROCm version we have available. To check the ROCm version run:

```
cat $ROCM_PATH/.info/version
```

Two minor versions before or after the current ROCm level should work.

Let's activate our environment for PyTorch.
```
source $HOME/venv-pt/bin/activate
```
and check the available versions:
```
pip install --index-url https://download.pytorch.org/whl/ torch== |& grep -o '[^ ]*rocm[^ ]*'
pip install --index-url https://download.pytorch.org/whl/ torchvision== |& grep -o '[^ ]*rocm[^ ]*'
pip install --index-url https://download.pytorch.org/whl/ torchaudio== |& grep -o '[^ ]*rocm[^ ]*'
```
It should yield something like:
```
// torch
...
2.6.0+rocm6.1,
2.6.0+rocm6.2.4,
// torchvision
...
0.21.0+rocm6.1,
0.21.0+rocm6.2.4,
// torchaudio
...
2.6.0+rocm6.1,
2.6.0+rocm6.2.4,
```
If you do not see the ROCm version you have in your system, you can find additional wheels [`here`](https://repo.radeon.com/rocm/manylinux/). As of April 13th 2025, there is no wheel for ROCm 6.3.3 on the PyTorch website. Hence we'll do:
```
pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.3/ --no-cache-dir
```
Let's do a quick smoke test to check that PyTorch can detect all GPUs:
```
python3 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
```
On an MI250, you should see `I have this many devices: 8`

Next, let's install TensorFlow in its respective environment. First, we deactivate the current environment that we used for PyTorch, then we activate the one for TensorFlow:

```
deactivate
source $HOME/venv-tf/bin/activate
```
The latest wheel for TensorFlow with ROCm can be found [`here`](https://pypi.org/project/tensorflow-rocm/). As of April 13th 2025, the latest available wheel is `tensorflow-rocm 2.14.0.600`, as it is also confirmed by doing:
```
pip install tensorflow-rocm==
```
Therefore, we install with:
```
pip3 install tensorflow-rocm==2.15.1  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.3/ --no-cache-dir
```
Once again, let's do a quick smoke tests to see if TensorFlow detects the AMD GPUs:
```
python3 -c 'from tensorflow.python.client import device_lib ; device_lib.list_local_devices()'
```
On MI250, it should show something like this:
```
2025-04-14 13:48:21.243911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:0 with 63828 MB memory:  -> device: 0, name: AMD Instinct MI250X/MI250, pci bus id: 0000:29:00.0
2025-04-14 13:48:21.500504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:1 with 63828 MB memory:  -> device: 1, name: AMD Instinct MI250X/MI250, pci bus id: 0000:2c:00.0
2025-04-14 13:48:21.748008: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:2 with 63828 MB memory:  -> device: 2, name: AMD Instinct MI250X/MI250, pci bus id: 0000:2f:00.0
2025-04-14 13:48:21.994639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:3 with 63828 MB memory:  -> device: 3, name: AMD Instinct MI250X/MI250, pci bus id: 0000:32:00.0
2025-04-14 13:48:22.242978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:4 with 63828 MB memory:  -> device: 4, name: AMD Instinct MI250X/MI250, pci bus id: 0000:ad:00.0
2025-04-14 13:48:22.489896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:5 with 63828 MB memory:  -> device: 5, name: AMD Instinct MI250X/MI250, pci bus id: 0000:b0:00.0
2025-04-14 13:48:22.736439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:6 with 63828 MB memory:  -> device: 6, name: AMD Instinct MI250X/MI250, pci bus id: 0000:b3:00.0
2025-04-14 13:48:22.982904: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /device:GPU:7 with 63828 MB memory:  -> device: 7, name: AMD Instinct MI250X/MI250, pci bus id: 0000:b6:00.0
```
We are interested in using Horovod with TensorFlow, so let's install it. Horovod build system was not made ready to ROCm 6.0+, so we need to provide some help to identify the new location for the Cmake files:
```
mkdir -p $HOME/cmake
cat > $HOME/cmake/cmake << EOF
#!/bin/bash -e

if [[ "\$@" == *"--build"* ]] ; then
  $(which cmake) \$@
else
  $(which cmake) -DCMAKE_MODULE_PATH=$ROCM_PATH/lib/cmake/hip \$@
fi
EOF
chmod +x $HOME/cmake/cmake
```

We can now build using our tuned cmake script:
```
module load rocm
module load openmpi

PATH=$HOME/cmake:$PATH \
CPATH=$ROCM_PATH/include/rccl \
HOROVOD_WITHOUT_MXNET=1 \
  HOROVOD_WITHOUT_GLOO=1 \
  HOROVOD_GPU=ROCM \
  HOROVOD_ROCM_HOME=$ROCM_PATH \
  HOROVOD_GPU_OPERATIONS=NCCL \
  HOROVOD_CPU_OPERATIONS=MPI \
  HOROVOD_WITH_MPI=1   \
  HOROVOD_ROCM_PATH=$ROCM_PATH \
  HOROVOD_RCCL_HOME=$ROCM_PATH/include/rccl \
  HOROVOD_RCCL_LIB=$ROCM_PATH/lib \
  HCC_AMDGPU_TARGET=gfx90a,gfx942 \
  HOROVOD_WITH_TENSORFLOW=1 \
  HOROVOD_WITHOUT_PYTORCH=1 \
  pip install --no-cache-dir --force-reinstall --verbose horovod==0.28.1
```
Let's define a work directory for us to try some examples.
```
mkdir -p $HOME/ai-with-rocm
```
## PyTorch MNIST example

MNIST is a quite popular data set for computer vision training. We are fortunate that are many examples on the internet on how to train MNIST dataset and they can usually be run without any changes.

Let's take one of the PyTorch official examples for this - we are training just two epochs:
```
cd $HOME/ai-with-rocm

deactivate
source $HOME/venv-pt/bin/activate

curl -LO https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py
module load rocm
python -u main.py --epochs 2 --batch-size 256
```
You may get an MIOpen error saying:

```
MIOpen(HIP): Error [FlushUnsafe] File is unwritable: "/tmp/gfx90a68.HIP.3_3_0_d22d5a13f-dirty.ufdb.txt"
```

In that case, the following commands should fix it:

```
mkdir -p $HOME/tmpdir
export TMPDIR=$HOME/tmpdir
rm -rf $HOME/tmpdir/*
echo "TMPDIR set to " $TMPDIR
```

We can control which GPU to use with the environmental variable ROCR_VISIBLE_DEVICES and can use the `rocprofv3` to get more of an idea about the GPU activity:

```
ROCR_VISIBLE_DEVICES=2 \
rocprofv3 --stats --kernel-trace -- python -u main.py --epochs 1 --batch-size 256
```
The resulting `*.csv` files show the different GPU kernels invoked for this application.

## PyTorch MNIST example - distributed

We might now be interested in distributing our training across devices. A way to accomplish this is by taking a distributed data-parallel (DDP) approach where each GPU will train different batch independently and combine the results afterwards.

There are also several examples on how to do this. We will use the following code as starting example:

```
cd $HOME/ai-with-rocm
curl -LO https://raw.githubusercontent.com/kubeflow/examples/master/pytorch_mnist/training/ddp/mnist/mnist_DDP.py
```

Download a modified version of this script and compare the differences:

```
curl -LO https://raw.githubusercontent.com/amd/HPCTrainingExamples/main/MLExamples/mnist_DDP_modified.py
vimdiff mnist_DDP.py mnist_DDP_modified.py
```

There are a couple of differences: one controls the batch size another one controls how the distributed run is initialized.
```
dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=int(os.environ['WORLD_SIZE']),                             rank=int(os.environ['RANK']))
```

PyTorch provides an object to control the distributed run environment:
```
import torch.distributed as dist
```
Here we are instructing that we want to use RCCL (AMD implementation for NCCL) and also want the tool to leverage the environment to collect more information, with some explicit information about the number of ranks (world size) and rank.

Other relevant bits to enable distributed run are in:
```
def run(modelpath, gpu):
  ...
  model = Net()
  ...
  model = torch.nn.parallel.DistributedDataParallel(model)
  ...
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
  ...
```
Here the model is wrapped into the `DistributedDataParallel` object to enable the distributed training.

Now to train the model on multiple ranks we will leverage MPI to start the processes and translate the MPI environment to something that PyTorch distributed package understands:

```
cat > run-me.sh << EOF
#!/bin/bash -e
  export MASTER_ADDR=localhost
  export MASTER_PORT=29500
  export WORLD_SIZE=\$OMPI_COMM_WORLD_SIZE
  export RANK=\$OMPI_COMM_WORLD_RANK
  export ROCR_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK

  python -u mnist_DDP_modified.py \
    --gpu --modelpath $HOME/ai-with-rocm/model

EOF
chmod +x run-me.sh
module load rocm
module load openmpi
mpirun -np 2 ./run-me.sh
```

Master address and port are defined for the different ranks to communicate between themselves. We then leverage the `OMPI_*` variables to decide ranks and GPUs to be used by each of these ranks.

Another popular way to spin a distributed run is to leverage the `torchrun` utility. However, this requires the application to include logic to decide which GPU to use, instead of relying on `ROCR_VISIBLE_DEVICES`. We can add the following line to our application after `dist.init_process_group()` (line 255 of `mnist_DDP_modified.py`) to accomplish that:

```
torch.cuda.set_device(int(os.environ['RANK']))
```
This will make GPUs to be indexed by the rank.

We can then run our application with `torchrun` as:
```
ROCR_VISIBLE_DEVICES=0,1 \
  torchrun --nnodes 1 --nproc_per_node 2 \
    ./mnist_DDP_modified.py --gpu --modelpath $HOME/ai-with-rocm/model
```

## TensorFlow with Horovod example

Similarly to PyTorch, TensorFlow examples should not need changes due to the GPU architecture.

Let's pick a TensorFlow example from the Horovod project - a synthectic training example already rigged to use Horovod:
```
deactivate
source $HOME/venv-tf/bin/activate
cd $HOME/ai-with-rocm

curl -LO https://raw.githubusercontent.com/horovod/horovod/master/examples/tensorflow2/tensorflow2_synthetic_benchmark.py

module load rocm
module load openmpi
mpirun -np 2 \
  python -u tensorflow2_synthetic_benchmark.py --batch-size 256
```
You should see an output similar to this one:
```
Running benchmark...
Iter #0: 559.4 img/sec per GPU
Iter #1: 557.7 img/sec per GPU
Iter #2: 559.2 img/sec per GPU
Iter #3: 552.1 img/sec per GPU
Iter #4: 553.6 img/sec per GPU
Iter #5: 551.0 img/sec per GPU
Iter #6: 550.9 img/sec per GPU
Iter #7: 553.8 img/sec per GPU
Iter #8: 553.4 img/sec per GPU
Iter #9: 546.8 img/sec per GPU
Img/sec per GPU: 553.8 +-7.4
Total img/sec on 2 GPU(s): 1107.6 +-14.9

```

If you encounter this runtime error:
```
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[1744644100.553280] [024ac696037d:188662:0]        ib_iface.c:1230 UCX  ERROR mlx5_0: ibv_create_cq(cqe=4096) failed: Cannot allocate memory : Please set max locked memory (ulimit -l) to 'unlimited' (current: 64 kbytes)
```

Then export the following variable to fix it:

```
export UCX_TLS=self,shm
```

Horovod makes it rather straightforward to start a distributed learning model as it leverages the already existing MPI environment.

We can inspect the test case and see the relevant bits where Horovod `hvd` is being leveraged, namely the selection of the GPU based on the local rank and wrapping of the local Gradient Tape operator: run
```
vi tensorflow2_synthetic_benchmark.py
```
And inspect the file, paying particular attention to the following lines of code:
```
import tensorflow as tf
import horovod.tensorflow as hvd
...
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
...
    with tf.GradientTape() as tape:
...
    tape = hvd.DistributedGradientTape(tape, compression=compression)
```

We can further inspect the GPU activity by leveraging `rocprofv3` to obtain a trace for one of the ranks. We'll do just 2 iterations so that the result files are not too large.
```
mpirun -np 2 \
  bash -c 'if [ $OMPI_COMM_WORLD_RANK -eq 1 ] ; then \
             profiler="rocprofv3 --hip-trace --output-format pftrace --" ; \
           fi ; \
           $profiler python -u tensorflow2_synthetic_benchmark.py \
             --batch-size 256 \
             --num-warmup-batches 2 \
             --num-iters 2'

```
A directory named with a hash will be created, inside of which the two trace files for each of the MPI processes will be available (for example):
```
193546_results.pftrace  193556_results.pftrace

```
We can also concatenate these trace files to be visualized in a single one:
```
 cat *.pftrace > perfetto_trace.pftrace
```
Then, we can visualize the trace file using the [`Perfetto`](https://www.ui.perfetto.dev/) tool. It is often a good idea to compress the file to make it easier to copy to one's workstation.
```
xz -T8 -9 perfetto_trace.pftrace
```
We can then copy it and visualize in Perfetto. We can detect the kernels from the different libraries, like MIOpen, rocBLAS, Eigen as well as MLIR JIT kernels.
![image](https://hackmd.io/_uploads/B1SpcGyW0.png)

## Examples with Huggingface transformers

There are several repositories of examples. A popular one is the Huggingface transformers. These examples should just work without any modification specific to AMD GPUs.

We can install the transformer package from source as:
```
deactivate
source $HOME/venv-tf/bin/activate

git clone \
https://github.com/huggingface/transformers.git \
  $HOME/ai-with-rocm/transformers
cd $HOME/ai-with-rocm/transformers
pip3 install -e .
```

It is useful to point the implementation to a suitable place to store and cache information and datasets. That can be done by setting the environment variables:
```
export HF_HOME=$HOME/ai-with-rocm/hf-home
export HUGGINGFACE_HUB_CACHE=$HOME/ai-with-rocm/hf-cache
mkdir -p $HF_HOME $HUGGINGFACE_HUB_CACHE
```
We can now try the examples.

### Image classification
Let's look at the image classification one. We need first to install the example dependencies:
```
cd $HOME/ai-with-rocm/transformers/examples/pytorch/image-classification
pip3 install -r requirements.txt
pip3 install scikit-learn
pip3 install -U pillow
```
We are now ready to run the example. We'll be experimenting with using mixed precision (with BF16 datatypes) or not (the default):
```
for precision in '' '--bf16' ; do
  ROCR_VISIBLE_DEVICES=2 \
    python3 run_image_classification.py \
      --dataset_name beans \
      --label_column_name labels \
      --output_dir $HOME/ai-with-rocm/hf-output  \
      --overwrite_output_dir \
      --remove_unused_columns False \
      --do_train \
      --learning_rate 2e-5 \
      --num_train_epochs 2 \
      --per_device_train_batch_size 8 \
      --torch_compile True \
      --seed 1337 \
      $precision
done
```
Depending on what is bounding the performance it could be good idea to use mixed precision or not - so this is a fair experiment to do. This should yield results like (if b16/gpu is supported):

```
...
# No mixed precision
***** train metrics *****
  epoch                    =         2.0
  total_flos               = 149248978GF
  train_loss               =      0.4118
  train_runtime            =  0:01:17.52
  train_samples_per_second =      26.675
  train_steps_per_second   =       3.354
...
# With mixed precision (BF16)
***** train metrics *****
  epoch                    =         2.0
  total_flos               = 149248978GF
  train_loss               =      0.4126
  train_runtime            =  0:01:23.73
  train_samples_per_second =      24.698
  train_steps_per_second   =       3.105
```

### Language modeling
A growing set of applications for distributed learning is language modeling. A typical approach is to leverage an established model and fine tune it for one's needs. That's what the `question-answering` example does. We'll start our fine-tuning from the BERT base dataset. The steps are similar to what we did before, first install the requirements:

```
cd $HOME/ai-with-rocm/transformers/examples/pytorch/question-answering
pip3 install -r requirements.txt
```
Then, we are ready to run the fine-tuning, e.g., on 2 GPUs for different training precisions:
```
for precision in '' '--bf16' '--fp16' ; do
ROCR_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node 2  run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_train \
    --per_device_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $HOME/ai-with-rocm/hf-output2  \
    --torch_compile True \
    --overwrite_output_dir \
    $precision
done
```



```
## FP32
***** train metrics *****
  epoch                    =        1.0
  total_flos               = 16159030GF
  train_loss               =     1.3039
  train_runtime            = 0:10:09.70
  train_samples            =      88524
  train_samples_per_second =    145.192
  train_steps_per_second   =      6.051

 # BF16
***** train metrics *****
  epoch                    =        1.0
  total_flos               = 16159030GF
  train_loss               =     1.3013
  train_runtime            = 0:07:34.81
  train_samples            =      88524
  train_samples_per_second =    194.636
  train_steps_per_second   =      8.111

 # FP16
 ***** train metrics *****
  epoch                    =        1.0
  total_flos               = 16159030GF
  train_loss               =     1.3064
  train_runtime            = 0:06:35.75
  train_samples            =      88524
  train_samples_per_second =    223.686
  train_steps_per_second   =      9.322
 ```

