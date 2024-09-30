# AMD Accelerator Cloud (AAC)
To support trainings, we can upload training containers to the AMD Accelerator Cloud (AAC), and have attendees login using the instructions below. This set of instructions assumes that users have already received their `<username>` and `<port_number>` for the container, and that they have either provided an ssh key to the training team, or they have received a password from the training team.

## Login Instructions
The instructions below rely on ssh to access the AAC. Remember that when a container is brought down, it will not be possible to access the user data on it, so make sure to backup your data frequently if you want to keep it.

### SSH-Key Generation
Generate an ssh key on your local system, which will be stored in `.ssh`:
```bash
cd $HOME
ssh-keygen -t ed25519 -N ''
```

To  examine the content of your public key do:
```bash
cat $HOME/.ssh/id_ed25519.pub
```

**NOTE**: at first login, you will be presented with the AAC user agreement form. This covers the terms of use of the compute hardware as well as how we will handle your data. Scroll down with the down arrow and type `yes` when prompted. Note that if you will scroll down too much, then `no` will be received as answer and you will be logged out.

### Login with SSH-Key

**IMPORTANT**: if you are supposed to login with an ssh key and you are prompted a password, do not type any password!
Instead, type `Ctrl+C` and contact us to let us know about the incident.

To login to AAC using the ssh key use the `<username>` and `<port_number>` that the training team has provided you, for instance:
```bash
ssh <username>@aac6.amd.com -i <path/to/ssh/key> -p <port_number> (1)
```

### Login with password
For a password login, the command is the same as in `(1)`, except that it is not necessary to specify a path to the ssh key. Just type the password that has been given to you when prompted:
```bash
ssh <username>@aac6.amd.com -p <port_number> 
```
**IMPORTANT**: It is fundamental to not type the wrong password more than two times otherwise your I.P. address will be blacklisted and you will not be allowed access to AAC until we modify our firewall to get you back in. This is especially important if you are at an event where all the attendees are connecting to the same wireless network.

In the commands above, `-p` refers to the port number and `-i` points to the path of your ssh key. Note that different port numbers
will be associated with different containers on the AAC, and anytime a container is brought up, the port number will change in general.
 
To simplify the login even further, you can add the following to your `.ssh/config` file:

```bash
# AMD AAC cluster
Host aac
   User <username>
   Hostname aac6.amd.com // this may be different depending on the container
   IdentityFile <path/to/ssh/key>
   Port <port_number>
   ServerAliveInterval 600
   ServerAliveCountMax 30
```
The `ServerAlive*` lines in the config file may be added to avoid timeouts when idle. you can then login using:
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

1. Check the spelling of the command ssh, in particular `<username>`, ` <port_number>` and password. 
2. Turn off VPN if on.
3. Try logging in from a different machine if available (and migrate the ssh key to the new machine or generate a new one and send it to us).
4. Try a ***jump host***: this is a local server that you ssh to and then do a second ssh command from there.

In case none of these options work, send us the output of the ssh command followed by `-vv` and also the output of `traceroute aac6.amd.com`. Additionally, let us know if the command `ping aac1.amd.com` works on your end.


### Directories and Files

Persistent storage is at `/datasets/teams/hackathon-testing/<username>`. Your home directory will be set to this directory: 

```bash
$HOME=/datasets/teams/hackathon-testing/<username>
```
Files in the above directory will persist across container starts and stops and even be available from another container with the same `<username>` on systems at the same hosting location. Remember that it will not be possible to retrieve your data once the container has been brought down.

You can copy files in or out of AAC with the `scp` or the `rsync` command.

Copy into AAC from your local system, for instance:

```bash
scp -i <path/to/ssh/key> -P <port_number> <file> <username>@aac6.amd.com:~/<path/to/file>
```
Copy from AAC to your local system:

```bash
scp -i <path/to/ssh/key> -P <port_number> <username>@aac6.amd.com:~/<path/to/file> .
```

To copy files in or out of the container, you can also use `rsync` as shown below:
```bash
rsync -avz -e "ssh -i <path/to/ssh/key> -p <port_number>" <file> <username>@aac6.amd.com:~/<path/to/file>
```

## Container Environment

Please consult the container's [README](https://github.com/amd/HPCTrainingDock/blob/main/README.md) to learn about the latest specs of the training container.

The container is based on the Ubuntu 22.04 Operating System with the ROCm 6.2.1 software stack. It contains multiple versions of AMD, GCC, and LLVM compilers, hip libraries, GPU-Aware MPI, AMD profiling tools and HPC community tools. The container also has modules set up with the lua modules package and a slurm package and configuration. It includes the following additional packages:

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
----------------------------------------------------------------------- /etc/lmod/modules/Linux ------------------------------------------------------------------------
   clang/base    clang/14 (D)    clang/15    gcc/base    gcc/11 (D)    gcc/12    gcc/13    miniconda3/23.11.0

------------------------------------------------------------------------ /etc/lmod/modules/ROCm ------------------------------------------------------------------------
   amdclang/17.0-6.2.1    hipfort/6.2.1    omniperf/6.2.1 (D)    omnitrace/6.2.1 (D)    opencl/6.2.1    rocm/6.2.1

-------------------------------------------------------------------- /etc/lmod/modules/ROCmPlus-MPI --------------------------------------------------------------------
   mpi4py/dev    mvapich/3.0    openmpi/5.0.5-ucc1.3.0-ucx1.17.0-xpmem2.7.3

------------------------------------------------------------- /etc/lmod/modules/ROCmPlus-AMDResearchTools --------------------------------------------------------------
   omniperf/2.0.0    omniperf/2.0.1    omnitrace/1.11.3

-------------------------------------------------------------- /etc/lmod/modules/ROCmPlus-LatestCompilers --------------------------------------------------------------
   amd-gcc/13.2.0    aomp/amdclang-19.0

-------------------------------------------------------------------- /etc/lmod/modules/ROCmPlus-AI ---------------------------------------------------------------------
   cupy/13.0.0b1    jax/0.4.32.dev    pytorch/2.4

------------------------------------------------------------------------ /etc/lmod/modules/misc ------------------------------------------------------------------------
   hpctoolkit/dev    kokkos/4.4.0    tau/dev

Where:   D:  Default Module
```
There are three modules associated with each ROCm version. One is the ROCm module which is needed by many of the other modules. The second is the amdclang module when using the amdclang compiler that comes bundled with ROCm. The third is the hipfort module for the Fortran interfaces to HIP.

Compiler modules set the C, CXX, FC flags. Note, only one compiler module can be loaded at a time. hipcc is in the path when the rocm module is loaded. 

## Slurm Information

The training container comes equipped with Slurm. Slurm configuration is for a single queue that is shared with the rest of the node. Run the following command to get info on Slurm: 

```bash
sinfo 
```

| PARTITION | AVAIL  | TIMELIMIT | NODES | STATE | NODELIST   |
|-----------|--------|-----------|-------|-------|------------|
| LocalQ    |  up    |	2:00:00  | 1     | idle  | localhost  |

The Slurm `salloc` command may be used to acquire a long term session that exclusively grants access to one or more GPUs. Alternatively, the `srun` or `sbatch` commands may be used to acquire a session with one or more GPUs and only exclusively use the session for the life of the run of an application. `squeue` will show information on who is currently running jobs.

### Training Examples Repo

The examples can also be obtained from our repo, which contains all the code that we will use for the exercises discussed during the training. To clone the repo, do:
```bash
cd $HOME
git clone https://github.com/amd/HPCTrainingExamples.git
```

