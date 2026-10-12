- [Prerequisites](#orgdb4878e)
  - [Start an interactive job](#org5a11c71)
    - [Hunter](#org3e225e2)
  - [Load the required modules](#org64836ed)
    - [Hunter (valid as of 2026-10-04)](#org05ff9cb)

Hands-on labs for the debugging course, split by part:

-   `cpu/` - **Part I** (Basics of gdb, CPU). Build the shared programs in `cpu/common/` once, then each `cpu/NN_name/` is an independent gdb lab.
-   `gpu/` - **Part II** (Basics of rocgdb, GPU). Each `gpu/NN_name/` is self-contained - its own sources and build line live in its `README.org`; there is no shared `common/`.


<a id="orgdb4878e"></a>

# Prerequisites

Get onto a node with the right toolchain first. The GPU labs need an actual AMD GPU; the CPU labs only need `gdb` (or `rocgdb`, which is a superset).


<a id="org5a11c71"></a>

## Start an interactive job

The login node is not for running jobs and usually has no GPU, so allocate a short interactive job. (Examples - use your site's own options.)


<a id="org3e225e2"></a>

### Hunter

```sh
# a 20-minute interactive job on one MI300A node
qsub -l select=1:node_type=mi300a -l walltime=0:20:00 -I -v TERM -q R_debug
```


<a id="org64836ed"></a>

## Load the required modules


<a id="org05ff9cb"></a>

### Hunter (valid as of 2026-10-04)

```sh
module unload cray-mpich cce
module use /opt/hlrs/testing/modulefiles/rocm-7.14/
module load mpich
module load cray-python # needed for CPU exercise 06_mdb on Hunter
```
