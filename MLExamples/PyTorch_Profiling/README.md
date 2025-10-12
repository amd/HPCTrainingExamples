
# AMD Profiling Tools for Python Machine Learning Applications

README.md from `HPCTrainingExamples/MLExamples/PyTorch_Profiling` in the Training Examples repository

This repository provides an overview of the tools available for profiling pytorch applications on AMD GPUs. The scripts used to reproduce some examples are available within this repository.

Some tools described below are AMD products, while others are community-based, open-source products supported by AMD on Instinct graphics cards.

## Profiling Methodology

When profiling a machine learning application in pytorch, the recommended tools and methodology is dictated by the goals. This document will cover techniques used to address specific profiling goals. One often needs to rely on multiple tools to obtain multiple views of an application, exposing more hotspots and bottlenecks to the developer.

In general, profiling of AI workloads is complicated by the many dynamic library dependencies and by dual language nature of the workloads: codes are orchestrated in python while relying on highly optimized C/C++ libraries that are loaded and executed at runtime. Since AI model developers rarely have the time and expertise to instrument and optimize vendor-developed low-level software libraries such as rocBLAS and hipBLASLt, performance profiling tools for AI workloads are highly focused on sampling-based techniques.

Unlike traditional HPC applications, AI applications rely on many kernels for various GEMM-based operations and tensor operations (normalizations, activations, etc.). Training workloads also require similar yet distinct kernels to perform backward passes for gradient calculations, as well as communication/computation overlapped collectives for synchronizing weights, gradients, and model updates. AI applications also target accelerated linear algebra operations in reduced precision. Compared to the traditional profiler use cases in HPC, where users may focus on just a few high-impact kernels, AI profiling and optimization requires a holistic, multifaceted profiling and optimization approach.

Profiling of AI applications typically proceeds as a top-down workflow with multiple stages:
1. Viewing workload performance at a system level such as via rocm-smi or amd-smi.
2. Investigating performance at a platform level with application timelines via built in tools like torch.profile and sampling profilers like ROCm System Profiler (formerly Omnitrace).
3. Investigating individual kernels or hotspots from compute, memory, cache behavior, or other properties via tools like RocProfiler (rocprofv3) or ROCm Compute Profiler (formerly Omniperf).


## Tools Overview

The following tools are available for profiling applications. Examples of all tools are provided in this repository for the included workload. The example tools and configurations are designed to be easily transferred to other workloads that are more interesting for profiling for the users.

- Pytorch profiler is an open source pytorch-level profiler available directly from pytorch (built-in to pytorch itself). This document does not cover the pytorch profiler, however it is supported natively on Instinct GPUs, e.g., AMD MI300X.

- RocProfiler is the default profiling tool that ships with ROCm and is available by default. RocProfiler provides ability to collect GPU kernel execution statistics as well as application GPU trace information. RocProfiler has undergone significant development and enhancement and the current best-known configuration is to use rocprofv3.

- ROCm Systems Profiler (formerly Omnitrace) is an enhanced tracing tool that can work with both instrumented code as well as provide sampling-based trace collection and profiling. It focussed on system level trace collection and shows both GPU and CPU activity on a same timeline.

- ROCm Compute Profiler (formerly Omniperf) is a GPU kernel-focus profiling and analysis tool. It is the tool of choice for easily accessing high-level analysis based on hardware counters, though some features for MI300X are available only starting with ROCm version 6.4.


## Organization of this Repository


This repository contains a simple pytorch training script that will train a classifier on CIFAR-100, and a collection of example scripts for the aforementioned profiling tools. The selected AI training example is simple on purpose, as the focus of this repository is to demonstrate how to use profiling tools with pytorch applications. Model is unoptimized to make performance problems easier to spot, and to the reader to potentially implement improvements.

Each profiling tool in a sub-directory has a dedicated README to elaborate on the scripts, profiling behavior, analysis tools and the expected outputs. It is meant to be a springboard for developers to quickly start profiling their workloads, while providing links to further documentation for detailed investigations.


## Workload Description

The workload in the `train_cifar_100.py` script is simple: this workload will train some number of iterations of a vision model to classify against the cifar100 dataset.  Data is provided from `torchvision`, while models are enabled through the `transformers` package. In the future, we hope to add additional datasets and models to encompass a broader range of results. Currently available configuration options are:

```bash
usage: train_cifar_100.py [-h] [--data-path DATA_PATH] [--batch-size BATCH_SIZE] [--download-only]
                          [--precision {float32,automixed,bfloat16}] [--max-epochs MAX_EPOCHS] [--max-steps MAX_STEPS]
                          [--torch-profile] [--model {resnet,swinv2,vit}]

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH, -dp DATA_PATH
                        Top level data storage
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size per rank
  --download-only
  --precision {float32,automixed,bfloat16}
  --max-epochs MAX_EPOCHS
                        Number of epochs (maximum) to run. Ignored if max_steps is set and is reached first
  --max-steps MAX_STEPS, -ms MAX_STEPS
                        Maximum number of steps to run for profiling
  --torch-profile       Activate the pytorch profiler
  --model {resnet,swinv2,vit}
                        Vision classification model to use
```

In the current workload, users may configure the datatype used (float32, bfloat16, or automatic fp16 with grad scaling) as well as the batch size and maximum number of iterations. By default, the maximum number of iterations is limited to 20 iterations to keep profiling runs short yet representative.

## Dataset Access

The datasets are available from `torchvision`, and the script `./download-data.sh` can be run once on a node with a GPU to download the data. By default, the data will be stored in a `data/` folder, and this is also the path where other scripts in this repo will search for the input data. You can also run `python3 train_cifar_100.py --download-only`, and if needed configure the data path using `--data-path [path]`.

## Helper Scripts

Each exercise folder contains bash scripts that wrap the main `train_cifar_100.py` with the appropriate setup and profiling commands. These scripts are meant to be used without containers (baremetal).

Single process scripts are the simplest, as they are using a single GPU to show the basic profiling functionallities. MPI scripts are similar, just demonstrating the usage for parallel, multi-process executions. Finally, `Slurm` scripts can help launch the jobs with the `Slurm` job scheduler environment. If you are using `Slurm` scripts, make sure that the `Slurm` config options match the ones available on your system (e.g., `--partition`).

Although these bash scripts should work on most system setups without any modifications, we encourage readers to carefully study them, especially the last commands which is profiling the execution.

<!-- The scripts in this repository are meant to be used in the `slurm` job scheduler environment, without containers (baremetal). Make sure that the slurm config options match the ones available on your system (e.g., `--partition`). Each script has available a "single process" configuration (same algorithm but not using more than one GPU) to show the basic profiling tool usage. -->

<!-- There are also `MPI` examples, and corresponding single-process scripts where appropriate.  For the `MPI` cases, the scripts assume OpenMPI and corresponding environment variables, which are used to initialize pytorch. -->

## Running the Scripts

This repository is meant as a development repository for profiling examples.  To simplify portability, please export this environment variable at the top level of the repository:

```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MLExamples/PyTorch_Profiling/
export PROFILER_TOP_DIR=$PWD
```

The scripts will expect the variables `MASTER_ADDR`, `MASTER_PORT`, and either `SLURM_PROCID`/`SLURM_NPROCS` or `OMPI_COMM_WORLD_RANK`/`OMPI_COMM_WORLD_SIZE` to configure the distributed process group. If these variables are not specified, scripts will assume certain default which may or may not work on the machine reader is using for testing. For single node tests, it will emit a warning if it does not find all variables, while multinode tests will run incorrectly without all.

Additionally, depending on the cluster environment, configure the `setup.sh` to load python, pytorch, and any other necessary tools unique to your environment.

## Development of Tools

AMD is continuing to update and improve profiling tools, both AMD-built tools as well as integrations into community tools. As these tools develop and continue to mature, users will find that sometimes arguments and configurations will change. We will strive to keep these examples up to date, but please open issues for any broken or outdated instructions. 



