
\newpage

# AMD Accelerator Cloud (AAC)

File: [login_info/AAC/README.md](`https://raw.githubusercontent.com/amd/HPCTrainingExamples/refs/heads/main/login_info/AAC/README.md`) at https://github.com/amd/HPCTrainingExamples

We have some small cloud based systems available for training activities. Attendees can login using the instructions 
below. This set of instructions assumes that users have already received their `<username>` and `<port_number>` for 
the container, and that they have either provided an ssh key to the training team, or they have received a password from the training team.

## Login Instructions
The instructions below rely on ssh to access the AAC. If you have not sent your public key in for an account and do not have an ssh public
key, start with the instructions on how to generate an ssh key. If you have sent an ssh key and received your account information, skip to
the section on how to log into the system. 

### SSH-Key Generation
Generate an ssh key on your local system, which will be stored in `.ssh`:
```bash
cd $HOME
ssh-keygen -t ed25519
```

To  examine the content of your public key do:
```bash
cat $HOME/.ssh/id_ed25519.pub
```

**NOTE**: at first login, you will be presented with the AAC user agreement form. This covers the terms of use of the compute hardware as well as how we will handle your data. Scroll down with the down arrow and type `yes` when prompted. Note that if you will scroll down too much, then `no` will be received as answer and you will be logged out.

### Login with SSH-Key

**IMPORTANT**: if you are supposed to login with an ssh key and you are prompted a password, do not type any password!
Instead, type `Ctrl+C` and contact us to get some help.

To login to an AAC MI300A system using the ssh key use the `<username>` that the training team has provided you, for instance:
```bash
ssh <username>@aac6.amd.com -i <path/to/ssh/key> (1)
```

### Login with password
For a password login, the command is the same as in `(1)`, except that it is not necessary to specify a path to the ssh key. Just type the password that has been given to you when prompted:
```bash
ssh <username>@aac6.amd.com 
```
**IMPORTANT**: It is fundamental to not type the wrong password more than two times otherwise your I.P. address will be blacklisted and you will not be allowed access to AAC until we modify our firewall to get you back in. This is especially important if you are at an event where all the attendees are connecting to the same wireless network.

If you are using a password login, you can upload an ssh key with the following command to avoid using a password

```bash
ssh-copy-id -i <path/to/ssh/key.pub> -o UpdateHostKeys=yes <username>@aac6.amd.com
```

In the commands above `-i` points to the path of your ssh key. The `-i` option is not needed if your
default key is used. 

 
To simplify the login even further, you can add the following to your `.ssh/config` file:

```bash
# AMD AAC cluster
Host aac
   User <username>
   Hostname aac6.amd.com // this may be different depending on the container
   IdentityFile <path/to/ssh/key> // this points to the private key file
   ServerAliveInterval 600
   ServerAliveCountMax 30
```
The `ServerAlive*` lines in the config file may be added to avoid timeouts when idle. You can then login using:
```bash
ssh aac -p <port_number>
```
It may also happen that a message like the following will show after logging into AAC:
```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
```
In such a case, remove in your local system the offending keys located in `.ssh/known_hosts`, as indicated by the warning message.

### Login Troubleshooting
Here are some troubleshooting tips if you cannot login to AAC following the instructions above:

1. Check the spelling of the command ssh, in particular `<username>` and password. 
2. Turn off VPN if on.
3. Try logging in from a different machine if available (and migrate the ssh key to the new machine or generate a new one and send it to us).
4. Try a ***jump host***: this is a local server that you ssh to and then do a second ssh command from there.

In case none of these options work, send us the output of the ssh command followed by `-vv` and also 
the output of `traceroute aac6.amd.com`. Additionally, let us know if the command `ping aac6.amd.com` works on your end.


### Directories and Files

```bash
$HOME=/home/aac/shared/teams/hackathon-testing/<group>/<username>`
```
You can copy files in or out of AAC with the `scp` or the `rsync` command.

Copy into AAC from your local system, for instance:

```bash
scp -i <path/to/ssh/key> <file> <username>@aac6.amd.com:~/<path/to/file>
```
Copy from AAC to your local system:

```bash
scp -i <path/to/ssh/key> <username>@aac6.amd.com:~/<path/to/file> .
```

To copy files in or out of the container, you can also use `rsync` as shown below:
```bash
rsync -avz -e "ssh -i <path/to/ssh/key>" <file> <username>@aac6.amd.com:~/<path/to/file>
```

## Container Environment

Please consult the container's [README](https://github.com/amd/HPCTrainingDock/blob/main/README.md) to learn about the latest specs of the training container.

The software on the node is based on the Ubuntu 22.04 Operating System with one of the latest versions of the 
ROCm software stack. It contains multiple versions of AMD, GCC, and LLVM compilers, 
hip libraries, GPU-Aware MPI, AMD profiling tools and HPC community tools. The container 
also has modules set up with the lua modules package and a slurm package and configuration. It includes the following additional packages:

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

### Explore Modules

To see what modules are available do:
```bash
module avail 
```

The output list of `module avail` should show:

```
--------------------------------- /etc/lmod/modules/Linux ----------------------------------
   clang/base    gcc/base

------------------------------- /etc/lmod/modules/LinuxPlus --------------------------------
   miniconda3/25.3.1    miniforge3/24.9.0

---------------------------------- /etc/lmod/modules/ROCm ----------------------------------
   amdclang/19.0.0-6.4.1           rocprof-tracedecoder/6.4.1
   amdflang-new/rocm-afar-7.0.5    rocprofiler-compute/6.4.1  (D)
   hipfort/6.4.1                   rocprofiler-sdk/6.4.1
   opencl/6.4.1                    rocprofiler-systems/6.4.1  (D)
   rocm/6.4.1

-------------------------------- /etc/lmod/modules/ROCmPlus --------------------------------
   adios2/2.10.1    hpctoolkit/2024.01.99-next    netcdf-c/4.9.3          scorep/9.0
   fftw/3.3.10      hypre/2.33.0                  netcdf-fortran/4.6.2    tau/dev
   hdf5/1.14.6      kokkos/4.6.01                 petsc/3.23.1

------------------------------ /etc/lmod/modules/ROCmPlus-MPI ------------------------------
   mpi4py/4.0.3    openmpi/5.0.7-ucc1.4.4-ucx1.18.1

----------------------- /etc/lmod/modules/ROCmPlus-AMDResearchTools ------------------------
   rocprofiler-compute/develop    rocprofiler-systems/amd-staging

------------------------ /etc/lmod/modules/ROCmPlus-LatestCompilers ------------------------
   hipfort_from_source/6.4.1

------------------------------ /etc/lmod/modules/ROCmPlus-AI -------------------------------
   cupy/14.0.0a1    jax/0.6.0                          pytorch/2.7.1           (D)
   ftorch/dev       pytorch/2.7.1_tunableop_enabled    tensorflow/merge-250318

---------------------------------- /etc/lmod/modules/misc ----------------------------------
   hipifly/dev

  Where:
   D:  Default Module
```

There are several modules associated with each ROCm version. One is the rocm module which is needed by many of the other modules. The second is the amdclang module when using the amdclang compiler that comes bundled with ROCm. The third is the hipfort module for the Fortran interfaces to HIP. Also, there is an OpenCL module and one for each of the AMD profilers.

Compiler modules set the C, CXX, FC flags. Only one compiler module can be loaded at a time. hipcc is in the path when the rocm module is loaded. Note that there are several modules that set the compiler flags and that they set the full path to the compilers to avoid path problems.

## Slurm Information

The AAC6 node is set up with Slurm. Slurm configuration is for a single queue that is shared with the rest of the node. Run the following command to get info on Slurm: 

```bash
sinfo 
```

```
PARTITION                    AVAIL  TIMELIMIT  NODES  STATE NODELIST
1CN192C4G1H_MI300A_Ubuntu22*    up 8-00:00:00      3   idle ppac-pl1-s24-[16,26,30,35],ppac-pl1-s25-40
1CN48C1G1H_MI300A_Ubuntu22      up 8-00:00:00      4   idle sh5-pl1-s12-[09,12,15,33,36]
```

The Slurm `salloc` command may be used to acquire a long term session that exclusively grants access to one or more GPUs. Alternatively, the `srun` or `sbatch` commands may be used to acquire a session with one or more GPUs and only exclusively use the session for the life of the run of an application. `squeue` will show information on who is currently running jobs.

## Training Examples Repo

You can get the examples from our repository.
This repository contains all the code that we normally use during our training events: 
```bash
cd $HOME
git clone https://github.com/amd/HPCTrainingExamples.git
```
