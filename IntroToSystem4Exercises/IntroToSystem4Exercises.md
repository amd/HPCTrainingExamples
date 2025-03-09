\pagebreak{}

# DiRAC Pre-Hackathon Workshop

## Basics

### SSH-Key Generation
Generate SSH key as shown below
```
cd $HOME
ssh-keygen -t ed25519 -N ''
cat $HOME/.ssh/id_ed25519.pub
```

### Logging in

```console
ssh <username>@aac1.amd.com -i id_ed25519 -p #### 
```
The port number will be established when the container environment starts up and will be given out at that time. 

At first login, you will be presented with the AMD Accelerator Cloud use agreement form. It covers the terms of use of the compute hardware as well as how we will handle your data.

To simplify the login even further, you can add the following to your `.ssh/config` file.

The `ServerAlive*` lines in the config file may be added to avoid timeouts when idle.
```bash
# AMD AAC cluster
Host aac
   User <USERNAME>
   Hostname aac1.amd.com
   IdentityFile id_ed25519 -- can put full path as well -- $HOME/.ssh/id_ed25519
   Port #### -- but this might change, so add on the command line shown below
   ServerAliveInterval 600
   ServerAliveCountMax 30
```
and then login using
```bash
ssh aac -p ####
```
### Directories and Files

Persistent storage is at `/home/aac/shared/teams/dcgpu_training/<groupid>/<userid>`. Your home directory will be set to this directory. 

```bash
$HOME=/home/aac/shared/teams/dcgpu_training/<groupid>/<userid>
```
Files in that directory will persist across container starts and stops and even be available from another container with the same userid on systems at the same hosting location.

You can copy files in or out with the `scp` or the `rsync` command.

Copy into AAC from your local system

```bash
scp -i <path>/<keyfile> -P #### <file> USER@aac1.amd.com:~/<path>/<file>
```
Copy from AAC to your local system

```bash
scp -i <path>/<keyfile> -P #### USER@aac1.amd.com:~/path/to/your/file ./
```

To copy files in or out of the container, you can also use `rsync` as shown below:
```bash
rsync -avz -e "ssh -i <path>/<keyfile> -p ####" <file> <USER>@aac1.amd.com:~/path/to/your/files
```

## Explore environment

This container is based on the Ubuntu 22.04 Operating System with the ROCm 6.3.3 software stack. It contains multiple versions of AMD, GCC, and LLVM compilers, hip libraries, GPU-Aware MPI (OpenMPI and MVAPICH), and AMD Profiling tools with perfetto and graphana. The container also has modules set up with the
lua modules package and a slurm package and configuration. It includes the following additional packages:

- emacs
- vim
- autotools
- cmake
- tmux
- boost
- eigen
- fftw
- gmp
- gsl
- hdf5-openmpi
- lapack
- magma
- matplotlib
- parmetis
- mpfr
- mpi4py
- openblas
- openssl
- swig
- numpy
- scipy
- h5sparse

### Check modules available:

```bash
module avail 
```

Output list of modules avail command

```
--------------------------------------------- /etc/lmod/modules/Linux ----------------------------------------------
   clang/base        clang/15    gcc/11 (D)    gcc/13               miniforge3/24.9.0
   clang/14   (D)    gcc/base    gcc/12        miniconda3/24.9.2

---------------------------------------------- /etc/lmod/modules/ROCm ----------------------------------------------
   amdclang/18.0.0-6.3.3        opencl/6.3.3    rocprofiler-compute/6.3.3
   hipfort/6.3.3         (D)    rocm/6.3.3      rocprofiler-systems/6.3.3

------------------------------------------ /etc/lmod/modules/ROCmPlus-MPI ------------------------------------------
   mpi4py/4.0.1    openmpi/5.0.7-ucc1.3.0-ucx1.18.0

------------------------------------ /etc/lmod/modules/ROCmPlus-LatestCompilers ------------------------------------
   amdflang-new-beta-drop/rocm-afar-7110-drop-5.3.0    aomp/amdclang-19.0    hipfort/6.3.3

------------------------------------------ /etc/lmod/modules/ROCmPlus-AI -------------------------------------------
   cupy/14.0.0a1    jax/0.4.35    pytorch/2.6.0

---------------------------------------------- /etc/lmod/modules/misc ----------------------------------------------
   fftw/3.3.10    hipifly/dev                 kokkos/4.5.01         netcdf-fortran/4.6.2-rc1    tau/dev
   hdf5/1.14.5    hpctoolkit/2024.11.27dev    netcdf-c/4.9.3-rc1    scorep/9.0-dev

  Where:
   D:  Default Module
```
There are several modules associated with each ROCm version. One is the rocm module which is needed by many of the other modules. The second is the amdclang module when using the amdclang compiler that comes bundled with ROCm. The third is the hipfort module for the Fortran interfaces to HIP. Also, there is an OpenCL module and one for each of the AMD profilers.

Compiler modules set the C, CXX, FC flags. Only one compiler module can be loaded at a time. hipcc is in the path when the rocm module is loaded. Note that there are several modules that set the compiler flags and that they set the full path to the compilers to avoid path problems.

## Slurm

The SLURM configuration is for a single queue that is shared with the rest of the node. 

```bash
sinfo 
```

| PARTITION | AVAIL  | TIMELIMIT | NODES | STATE | NODELIST   |
|-----------|--------|-----------|-------|-------|------------|
| LocalQ    |  up    |	2:00:00  | 1     | idle  | localhost  |

The SLURM salloc command may be used to acquire a long term session that exclusively grants access to one or more GPUs. Alternatively, the srun or sbatch commands may be used to acquire a session with one or more GPUs and only exclusively use the session for the life of the run of an application. squeue and sinfo will show information about the current state of the SLURM system.

## Exercise examples

The exercise examples are preloaded into the /Shared directory. Copy the files into your home directory with 
```bash
mkdir -p $HOME/HPCTrainingExamples
scp -pr /Shared/HPCTrainingExamples/* $HOME/HPCTrainingExamples/
```

If you need to refer to the examples separately, they are at the repository below.

## Examples repo


Alternatively, you can get the examples from our repo.
This repo contains all the code that we will use for the exercises that follow
```bash
cd $HOME
git clone https://github.com/amd/HPCTrainingExamples.git
```

