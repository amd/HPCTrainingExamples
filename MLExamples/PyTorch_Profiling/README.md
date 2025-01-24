# AMD Profiling Tools for Python Machine Learning Applications

This repository provides an overview of the tools available for profiling applications on AMD GPUs, with a specific focus on applications using pytorch in a multinode, slurm-based environment.  The scripts used to reproduce some examples are available with this repository.

Some tools described below are specifically AMD products, while others are community-based, open source products supported by AMD on Instinct graphics cards.

## Profiling Methodology

When profiling a machine learning application in pytorch, the recommended tools and methodology is dictated by the goals.  This document will cover techniques used to address specific profiling goals.  Often, it is most useful to use multiple tools to obtain multiple views of an application, exposing more hotspots and bottlenecks to the developer.

In general, profiling of AI workloads is complicated by the many dynamic library dependencies and dual language nature of the workloads: codes are orchestrated in python and rely on highly optimized C/C++ libraries that are loaded and executed at runtime.  Because AI Model developers rarely have the time and expertise to instrument vendor-developed software libraries such as rocBLAS and hipBLASLt, performance profiling tools for AI workloads are highly focused on sampling-based techniques.

Unlike traditional HPC applications, AI applications rely on many kernels for various GEMM-based operations and tensor operations (normalizations, activations, et cetera).  Training workloads also require similar yet distinct kernels to perform backward passes for gradient calculations, as well as communication/computation overlapped collectives for synchronizing weights, gradients, and performing model updates.  AI applications also target accelerated linear algebra operations in reduced precision.  Compared to the traditional profiler use cases in HPC, where users may focus on just a few high-impact kernels, AI profiling and optimization requires a holistic, multifaceted profiling and optimization approach.

Profiling of AI applications typically proceeds as a top-down workflow in multiple stages, viewing workload performance at a system level (such as via omnitstat), investigating performance at a platform level with application timelines (via built in tools like torch.profile and sampling profilers like ROCm System Profiler - formerly Omnitrace), and investigating individual kernels or hotspots from compute, memory, cache behavior, or other properties (via tools like ROCm Compute Profiler - formerly Omniperf)


## Tools Overview

The following tools are available for profiling applications.  Examples of all tools are provided in this repository for the included workload.  The example tools and configurations are designed to be easily transferred to other, more interesting workloads for profiling.

- Pytorch profiler is an open source pytorch-level profiler available directly from pytorch (and built in to pytorch itself).  This document does not cover the pytorch profiler, however it is supported natively on, for instance, AMD MI300X.

- RocProfiler is the default profiling tool that ships with ROCm and is available by default.  RocProfiler provides ability to collect kernel execution statistics as well as application trace information.  RocProfiler has undergone significant development and enhancement and the current best known configuration is to use rocprofv3.

- ROCm Systems Profiler (formerly Omnitrace) is an enhanced tracing tool that can work with both instrumented code as well as provide sampling based trace collection and profiling.

- ROCm Compute Profiler (formerly Omniperf) is a kernel-focus profiling and analysis tool.  It is the tool of choice for easily accessing highlevel analysis based on hardware counters, though some features for MI300X will not be available until ROCm 6.4.




# Organization of this Repository


This repository is a simple pytorch training script that will train a classifier on CIFAR-100, and a collection of example scripts for the profiling tools mentioned above. The AI training here is not complicated, which is the point: this repository is meant to demonstrate how to use profiling tools with pytorch applications, so the model is kept simple - and unoptimized, to make problems easier to spot!

Each profiling tool in a sub directory has a dedicated README to elaborate on the scripts, profiling behavior, and analysis tools.  It is meant to be a spring board for developers to quickly start profiling their workloads, while providing links to further documentation for detailed investigations.


## Workload Description

The workload in the `train_cifar_100.py` script is simple: this workload will train some number of iterations of a vision model to classifiy against the cifar100 dataset.  Data is provided from `torchvision`, while models are enabled through the `transformers` package.  In the future, we hope to add additional datasets and models to encompass a broader range of results.  The current configuration options available are:

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
                        Number of epochs (maximum) to run. Ignored if max_steps is set and is reached first.
  --max-steps MAX_STEPS, -ms MAX_STEPS
                        Maximum number of steps to run for profiling
  --torch-profile       Activate the pytorch profiler
  --model {resnet,swinv2,vit}
                        Vision classification model to use
```

In the current workload, users may configure the datatype used (float32, bfloat16, or automatic fp16 with grad scaling) as well as the batch size and maximum number of iterations.  By default, the maximum number of iterations is limited to 20 iterations to keep profiling runs short but representative.

### Dataset Access

The datasets are available from `torchvision`, and the script can be run once in offline mode to download the data with `python train_cifar_100.py --download-only`.  The data path can be configured with `--data-path [path]`.

## Organization of the Scripts

The scripts in this repository are meant to be used in the `slurm` job scheduler environment, without containers (baremetal).  Each script has available a "single process" configuration (same algorithm, but not using more than one GPU) to show the basic profiling tool usage.

<!-- There are also `MPI` examples, and corresponding single-process scripts where appropriate.  For the `MPI` cases, the scripts assume OpenMPI and corresponding environment variables, which are used to initialize pytorch. -->

# Running the scripts

This repository is meant as a development repository for profiling examples.  To simplify portability, please export this environment variable at the top level of the repository:

```bash
# (git clone the repo...)
cd pytorch-profiling-examples
export PROFILER_TOP_DIR=$PWD
```

The scripts will expect the variables `MASTER_ADDR`, `MASTER_PORT`, and either `SLURM_PROCID`/`SLURM_NPROCS` or `OMPI_COMM_WORLD_RANK`/`OMPI_COMM_WORLD_SIZE` to configure the distributed process group.  For single node tests, it will emit a warning if it does not find all variables, while multinode tests will run incorrectly without all.

Additionally, depending on the cluster environment, configure the `setup.sh` to load python, pytorch, and any other necessary tools unique to your environment.

## Development of Tools

AMD is continuing to update and improve profiling tools, both AMD-built tools as well as integrations into community tools.  As these tools develop and continue to mature, users will find that sometimes arguments and configurations will change.  We will strive to keep these examples up to date, but please open issues for any broken or outdated instructions. 
