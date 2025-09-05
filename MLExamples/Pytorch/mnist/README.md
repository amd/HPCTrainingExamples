#MNIST pytorch examples

We'll cover three different ways to run Pytorch jobs.

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

## Virtual environment

Probably the most common way to run a pytorch application is to use a python
virtual environment. It is the most flexible and can be used from your local
PC, to a workstation and then for larger jobs, on an HPC system with the latest
powerful GPUs.

On an HPC cluster, we need to get an allocation with a GPU so that we can
run this exercise. We'll get just one GPU to begin with so that we don't
monopolize the resources.

```
salloc -p 1CN192C4G1H_MI300A_Ubuntu22 --ntasks 24 --gpus=1 --time=04:00:0
```

First create a virtual environment. This will provide an isolated working space that
will be separated from other python packages.

```
python3 -m venv rocm-pytorch
```

This will create a directory rocm-pytorch. We'll change to that
directory and activate the virtual environment.

```
cd rocm-pytorch
source bin/activate
```

This is an empty environment. We have to populate it with the 
software that we need. There is a python package containing 
ROCm and pytorch that we will use.

```
pip3 install rocm=pytorch:latest-release
```

There are other versions such as "latest". We want a more
reproducible version. Another option is to install a
numbered pytorch version for even more reproducibility.

Before we jump into running our project, lets confirm that
the environment is set up and running properly. We'll first 
confirm that we can import the torch module in our python
environment. Then we'll check that we can access a GPU.

```
pytorch -c 'import torch` | 
pytorch -c '
```

Now let's run the MNIST example

```
python3 main.py
```

Cleaning up after our run, we deactivate the virtual environment
and remove our directory. With the amount of disk space that these
environments can consume, it is important to clean up if we don't
plan to reuse it.

```
deactivate
cd ../../..
rm -rf rocm-pytorch
```

Now that we have it running, it is best to submit the project
to the HPC cluster using a batch script. Exit the allocation
and set up to submit the job using a batch script.

```
#SBATCH
```

And we submit the batch script with 

```
sbatch pytorch_mnist_venv.batch
```

The output will show up in pytorch_mnist_venv.out. Check the result and the
time it took to run the job. Also, confirm that it ran on the GPU.

If we wanted to reuse the virtual environment that we created. we could
leave the directory and just start with sourcing the activate script. This
would save some time and network traffic.

Multiple GPU runs

To change the script to run on multiple GPUs, we can follow the same
general procedure. First, we modify the batch script to get two GPUs.

```
```

Then we modify the main.py script to use the GPUs that we have allocated.
This is done right after line 127 by wrapping the ... But to do this, we
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

### Modifications to run on multiple GPUs

Now we can add the code to model to make it use multiple GPUs. The 
added lines are:

```
```

We have conveniently created a sed script that can be used to make
this modification on the fly for our Slurm batch script:

```
sed
```

And now we put this all together into a new batch script which we
can submit to run.

```
sbatch pytorch_mnist_venv_2gpus.batch
```

Output will be in `pytorch_mnist_venv_2gpus.out`

## Container Environment

We'll look at two different container systems for running this problem. HPC centers usually
support only non-root options for containers. Check with your local HPC documentation for
what they recommend.


### Apptainer

With Apptainer, we first need to download the container to the local system. Most container
images are in docker format, so we download it and convert it to the "sif" format. We can do this 
in advance and keep the converted image local for future runs. Check with your local system
on whether they recommend that this is done on the front-end or as a batch job. 

The container image will have a self-contained operating system, python and ROCm version.
So we can just get the latest version. You may also want to use the latest named version
to get more reproducible results.

```
apptainer pull rocm-pytorch.sif docker://rocm/pytorch:latest
```

Now we can follow similar steps as done in the previous example, but replace the pip install
with starting up the apptainer image. We get an GPU allocation with salloc and start up the 
container image with a shell.

```
salloc --ntasks 16 --gpus=1 --time=04:00:0``
apptainer shell --rocm rocm-pytorch.sif
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
python3 main.py
```

Exit the shell and the allocation and clean up.

#### Apptainer batch file

It is better to run as a batch job once you have the verified the steps that you need.
We put all the commands in a batch file. And we change the apptainer command to exec
instead of shell. Apptainer will run the command following the load of the container
image, so we put all our commands into a single set of double quotes and use the bash
command to execute it as a script in that shell. The batch file should look something
like the following. We'll name the file pytorch_mnist_apptainer.batch.

``` bash
#!/bin/bash
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --gpus=1
#SBATCH --time=05:00:0
#SBATCH --ntasks=4
#SBATCH --output=pytorch_mnist_apptainer.out
#SBATCH --error=pytorch_mnist_apptainer.err

apptainer exec --rocm rocm-pytorch.sif bash -c " \
   python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure' && \
   python3 -c 'import torch; print(torch.cuda.is_available())' && \
   pip3 install --user -r requirements.txt; \
   cd pytorch_examples/mnist && \
   python3 main.py && \
   echo 'Finished'"
```

We submit the job with 

```
sbatch pytorch_mnist_apptainer.batch
```

Output will be in `pytorch_mnist_apptainer.out`

### Podman

To run the same example with podman, we can directly use the docker container. The steps are


```
```

## Modules with local rocm and pytorch software

For sites that are heavily using pytorch, it may be worth installing it locally on the system
and providing it to users in a module environment. The local pytorch installation can be done
with pip installs or by building from source. Building from source allows customization and
optimization for the local CPU and GPU hardware. 

In the local pytorch module, the version is built from source with the GPU-aware MPI and only
the MI200 and MI300 series GPU gfx model support. The build is also done without miniconda
or miniforge so that it is self-standing and not dependent on another package.
Also, many additional pytorch packages have been 
added. These include

```
maybe numbered?
torchvision
torchaudio
flashattention
sageattention
transformers
```

To run the examples with a pytorch module

```
module load rocm pytorch
git clone
cd pytorch-examples/mnist
python3 main.py
```

If additional python packages need to be added, it is suggested to create a python virtual
environment as in the first example. Also, to fully isolate the pytorch module, it is recommended to
clear the PYTHONPATH variable. The pytorch module will add to the PYTHONPATH variable the paths
necessary for the software in the module. To support loading other module packages, the additions
are appended to the existing PYTHONPATH. 

## Wrapup

There are also many combinations of the approaches presented. There is not an ideal choice that will
work on every system. Consider how each works for your situation and the system you are running on. Consult
the recommendations for your HPC center for guidance.
What works best on your local system may not be the best in an HPC or cloud environment. Consider the
load on the system, especially network load and file system space.

Here is a table summarizing the pros and cons of each approach.

* Network load
* Filesystem space
* Runtime


