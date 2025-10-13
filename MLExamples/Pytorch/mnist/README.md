# MNIST pytorch examples

README.md from `HPCTrainingExamples/MLExamples/Pytorch/mnist` in the Training Examples repository

We'll cover three different ways to run Pytorch jobs. Why are
there different ways? Each method fits a different scenario
and system configuration. Knowing how to use each of them is
important since the method you use on your laptop or workstation
may not be ideal on an HPC system.

First we'll set up the problem and make a few small modifications.

```
git clone https://github.com/pytorch/examples.git pytorch_examples
cd pytorch_examples
```

The example we want to run is the MNIST training
example. It is a Image classification (MNIST) using Convnets.
We go to that directory and set up for our run.

```
cd mnist
```

The key file in this directory is main.py. This script will run on either CPUs or GPUs,
depending on what is available. For a GPU, the data and
the model need to be transferred to the GPU device. This
is done with the "to.device" commands. The device will
be defined as a CPU on a CPU and a GPU on a system with
a GPU that is available and can be used.

To enhance the example, we'll modify a line to confirm
that we are running on our AMD GPU and not the CPU.
Just after line 101, add a line to print out the device
being used. The line to be added to main.py is:

```
   print(f"Using device: {device}")' main.py`
```

For our scripts, we'll use a stream editor, sed, to add the
line on the fly.

```
sed -i -e '/device = torch.device("cpu")/a\    print(f"Using device: {device}")' main.py
```

And return to the base directory with

```
cd ../..
```

## Virtual environment

Probably the most common way to run a pytorch application is to use a python
virtual environment. It is the most flexible and can be used from your local
PC, to a workstation and then for larger jobs, on an HPC system with the latest
powerful GPUs.

On an HPC cluster, we need to get an allocation with a GPU so that we can
run this exercise. We'll get just one GPU to begin with so that we don't
monopolize the resources.

```
salloc --ntasks 1 --gpus=1 --time=01:00:00
```

Go to the directory with the mnist example

```
cd ~/pytorch_examples/mnist
```

Create a virtual environment. This will provide an isolated working space that
will be separated from other python packages.

```
python3 -m venv rocm-pytorch
source rocm-pytorch/bin/activate
```

This will create a directory rocm-pytorch.

This is an empty environment. We have to populate it with the 
software that we need. There is a python package containing 
ROCm and pytorch that we will use. We time the install for comparison.

```
time pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

This is for ROCm 6.4.x releases. If you have a different ROCm release, you
should modify it for your version.

Before we jump into running our project, lets confirm that
the environment is set up and running properly. We'll first 
confirm that we can import the torch module in our python
environment. Then we'll check that we can access a GPU.

```
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'
```

Now let's run the MNIST example. We time it for later analysis.

```
time python3 main.py
```

You should see a line at the start that says `Using device: cuda`. It
confirms that you are running on the GPU. It is the AMD GPU even
though it says cuda.

Cleaning up after our run, we deactivate the virtual environment
and remove our directory. With the amount of disk space that these
environments can consume, it is important to clean up if we don't
plan to reuse it.

```
deactivate
rm -rf rocm-pytorch
```

Be sure and exit the allocation to free it up for other users.

```
exit
```

Now that we have it running, it is best to submit the project
to the HPC cluster using a batch script. Exit the allocation
and set up to submit the job using a batch script. For convenience,
clone the HPC Training Examples and go to the directory with
the scripts. Or you can cut and paste the following into a batch script named
`pytorch_mnist_venv.batch`.

```
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/MLExamples/Pytorch/mnist
```

Here are the contents of the batch script.

```
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=pytorch_mnist_venv.out
#SBATCH --error=pytorch_mnist_venv.out

cd ~/pytorch_examples/mnist
python3 -m venv rocm-pytorch
source rocm-pytorch/bin/activate
echo "Starting pytorch install"
time pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'

echo "Starting mnist"
time python3 main.py

deactivate
rm -rf rocm-pytorch

echo "Finished"
```

And we submit the batch script with 

```
sbatch pytorch_mnist_venv.batch
```

The output will show up in `pytorch_mnist_venv.out`. Check the result and the
time it took to run the job. Also, confirm that it ran on the GPU.

If we wanted to reuse the virtual environment that we created. we could
leave the directory and just start with sourcing the activate script. This
would save some time and network traffic.


## Container Environment

We'll look at two different container systems for running this problem. HPC centers usually
support only non-root options for containers. Check with your local HPC documentation for
what they recommend.


### Apptainer

With Apptainer, we first need to download the container to the local system. Most container
images are in docker format, so we download it and convert it to the "sif" format. We can do this 
in advance and keep the converted image local for future runs. Check with your local system
on whether they recommend that this is done on the front-end or as a batch job. On this
system, we use the SH5 nodes with the single GPU so that we don't tie up the nodes
with more GPUs.

The container image will have a self-contained operating system, python and ROCm version.
We try to get the closest matches to our current system versions. Squashfs needs a lot of
memory, soe we'll get an exclusive allocation on one of the SH5 nodes.

```
salloc --exclusive  -p 1CN48C1G1H_MI300A_Ubuntu22
apptainer pull rocm-pytorch.sif docker://rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
exit
```

Now we can follow similar steps as done in the previous example, but replace the pip install
with starting up the apptainer image. We get an GPU allocation with salloc and start up the 
container image with a shell.

```
salloc --ntasks 1 --gpus=1 --time=01:00:00
apptainer shell --rocm ~/rocm-pytorch.sif
```

We can test whether the pytorch is functioning correctly and that we can access a GPU.

```
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'
```

We are ready to run the example problem. We go to the directory with the example files.

```
cd pytorch_examples/mnist
```

We need to install the python library requirements.

```
pip3 install --user -r requirements.txt
```

And we are finally ready to run the example.

```
time python3 main.py
```

Exit the shell and then exit the allocation and clean up.

#### Apptainer batch file

It is better to run as a batch job once you have the verified the steps that you need.
We put all the commands in a batch file. And we change the apptainer command to exec
instead of "shell". Apptainer will run the command following the load of the container
image. The batch file should look something
like the following. We'll name the file `pytorch_mnist_apptainer.batch`.

``` bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=pytorch_mnist_apptainer.out
#SBATCH --error=pytorch_mnist_apptainer.out

# Pre-pull rocm-pytorch image on a login/front-end node
#  apptainer pull rocm-pytorch.sif docker://rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.5.1

mkdir -p .torch

cd $HOME/pytorch_examples/mnist

apptainer exec --rocm \
   --bind "$PWD:/workspace" -W /workspace \
   --env TORCH_HOME=/workspace/.torch \
   ~/rocm-pytorch.sif  \
pwd
python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

pip3 install --user -r requirements.txt
time python3 main.py
```

We submit the job with 

```
sbatch pytorch_mnist_apptainer.batch
```

Output will be in `pytorch_mnist_apptainer.out`


### Podman

To run the same example with podman, we can directly use the docker container.
The steps for an interactive job are

First, get an allocation with a GPU

```
salloc --ntasks 1 --gpus=1 --time=01:00:00
```

Now start up an interactive shell in Podman

```
podman run -it --device=/dev/dri --device=/dev/kfd --network=host --ipc=host --userns=keep-id --group-add=keep-groups --cgroupns=host --cap-add=SYS_PTRACE --security-o
pt seccomp=unconfined -v $HOME:/workdir -w /workdir docker://rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0 bash
```

cd pytorch_examples/mnist
pip3 install -r requirements.txt

python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

time python3 main.py

```
podman pull docker://rocm/pytorch:latest
```

Then the Slurm batch file for the podman job is

```
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=pytorch_mnist_podman.out
#SBATCH --error=pytorch_mnist_podman.out

cd $HOME/pytorch_examples/mnist

podman run --rm \
  --device=/dev/dri --device=/dev/kfd --network=host --ipc=host \
  --userns=keep-id --group-add=keep-groups --cgroupns=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $PWD:/workdir -w /workdir \
  docker://rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0 \
pip3 install --user -r requirements.txt

python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

time python3 main.py
```

## Modules with local rocm and pytorch software

For sites that are heavily using pytorch, it may be worth installing it locally on the system
and providing it to users in a module environment. The local pytorch installation can be done
with pip installs or by building from source. Building from source allows customization and
optimization for the local CPU and GPU hardware. 

In this local pytorch module, the version is built from source with the GPU-aware MPI and only
the MI200 and MI300 series GPU gfx model support. The build is also done without miniconda
or miniforge so that it is self-standing and not dependent on another package.
Also, many additional pytorch packages have been 
added. These include

- torchvision
- torchaudio
- triton
- transformers
- sageattention
- flashattention

We get an interactive compute node with a single GPU

```
salloc --ntasks 1 --gpus=1 --time=01:00:00
```

Load the environment

```
module load rocm pytorch
```

Going to the mnist example

```
cd ~/pytorch_examples/mnist
pip3 install --user -r requirements.txt
```

Check that it is working properly.

```
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'
```

We can get more information with the same check script that we used above in the
container examples.

```
python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF
```

We can now move on to running our pytorch application

```
time python3 main.py
```

Exit the allocation

```
exit
```

If additional python packages need to be added, like above, it is suggested to create a python virtual
environment as in the first example. Also, to fully isolate the pytorch module, it is recommended to
clear the PYTHONPATH variable. The pytorch module will add to the PYTHONPATH variable the paths
necessary for the software in the module. To support loading other module packages, the additions
are appended to the existing PYTHONPATH. With these additions, our example looks like:

We get an interactive compute node with a single GPU

```
salloc --ntasks 1 --gpus=1 --time=01:00:00
```

Changing to the location of the mnist example and creating a virtual environment

```
module load rocm pytorch
cd ~/pytorch_examples/mnist
python3 -m venv rocm-pytorch
source rocm-pytorch/bin/activate
pip3 install -r requirements.txt

python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

time python3 main.py

deactivate
rm -rf rocm-pytorch
```

We can make this run as a batch file by adding the batch file directives
at the top. First without the virtual environment

```
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:0
#SBATCH --ntasks=48
#SBATCH --output=pytorch_mnist_module.out
#SBATCH --error=pytorch_mnist_module.out

module load rocm pytorch
python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

~/pytorch_examples/mnist
pip3 install --user -r requirements.txt
time python3 main.py
```

And submit the job with

```
sbatch pytorch_mnist_module.batch
```

With a virtual environment in the `pytorch_mnist_module_venv.batch` file

```
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:0
#SBATCH --ntasks=48
#SBATCH --output=pytorch_mnist_module_venv.out
#SBATCH --error=pytorch_mnist_module_venv.out

python3 -m venv rocm-pytorch
pushd rocm-pytorch
source bin/activate

module load rocm pytorch
python3 - << EOF
import torch, platform
print("Torch module location: ",torch)
print("Torch:", torch.__version__, " HIP:", getattr(torch.version, "hip", None))
print("Platform:", platform.platform())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF

pushd ~/pytorch_examples/mnist
pip3 install -r requirements.txt
python3 main.py

popd
deactivate
popd
rm -rf rocm-pytorch
```

And submit the job with

```
sbatch pytorch_mnist_module_venv.batch
```

### Modifications to run on multiple GPUs

It is a simple procedure to run on multiple GPUs.

We first modify the main.py script to use the GPUs that we have allocated.
This is done right after line 127 by wrapping the model in a DataParallel
block. But to do this, we
have to split the line setting the model from the operation sending it
to the device.

```
model = Net().to(device)
```

becomes

```
model = Net()
model.to(device)
```

Now we can add the code to `model` to make it use multiple GPUs. 
The revised and lines are:

```
model = Net()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
```

We have conveniently created a sed script that can be used to make
this modification on the fly for our Slurm batch script:

```
sed -i -e '/device = torch.device("cpu")/a\    print(f"Using device: {device}")' main.py
sed -i -e '/model = Net().to(device)/s/.to(device)//' main.py
sed -i -e '/model = Net()/a\    if torch.cuda.device_count() > 1:\
        print(f"Using {torch.cuda.device_count()} GPUs!")\
        model = nn.DataParallel(model)\
    model.to(device)' main.py
```

And now we make a new batch script with a simple change to the SBATCH 
directives to get more GPUs. 

```
#!/bin/bash
#SBATCH --gpus=2
#SBATCH --time=01:00:0
#SBATCH --ntasks=2
#SBATCH --output=pytorch_mnist_venv_2gpus.out
#SBATCH --error=pytorch_mnist_venv_2gpus.out

module load rocm pytorch

cd ~/pytorch_examples/mnist
python3 -m venv rocm-pytorch-2gpus
source rocm-pytorch-2gpus/bin/activate
echo "Starting pytorch install"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'

echo "Starting mnist"
time python3 main.py

deactivate
rm -rf rocm-pytorch-2gpus
```
All that is left to do is submit the job.

```
sbatch pytorch_mnist_venv_2gpus.batch
```

Output will be in `pytorch_mnist_venv_2gpus.out`

## Wrapup

There are also many combinations of the approaches presented. There is not one ideal choice that will
work on every system. Consider how each works for your situation and the system you are running on. Consult
the recommendations for your HPC center for guidance.
What works best on your local system may not be the best in an HPC or cloud environment. Consider the
load on the system, especially network load and file system space.

