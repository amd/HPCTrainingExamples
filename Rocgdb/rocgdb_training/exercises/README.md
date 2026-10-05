- [Prerequisites](#orgb359aee)
  - [Start an interactive job](#org1319956)
    - [Hunter](#orga7ebce0)
  - [Load the required modules](#orgcbccc93)
    - [Hunter (valid as of 2026-10-04)](#orga89a220)

Hands-on labs for the debugging course, split by part:

-   `cpu/`: Part I (Basics of gdb, CPU). Build the shared programs in `cpu/common/` once, then each `cpu/NN_name/` is an independent gdb lab.
-   `gpu/`: Part II (Basics of rocgdb, GPU). Each `gpu/NN_name/` is self-contained: its own sources and build line live in its `README.md`; there is no shared `common/`.


<a id="orgb359aee"></a>

# Prerequisites

Get onto a node with the right toolchain first. The GPU labs need an actual AMD GPU; the CPU labs only need `gdb` (or `rocgdb`, which is a superset).


<a id="org1319956"></a>

## Start an interactive job

The login node is not for running jobs and usually has no GPU, so allocate a short interactive job. (Examples: use your site's own options.)


<a id="orga7ebce0"></a>

### Hunter

```sh
# a 20-minute interactive job on one MI300A node
qsub -l select=1:node_type=mi300a -l walltime=0:20:00 -I
```


<a id="orgcbccc93"></a>

## Load the required modules


<a id="orga89a220"></a>

### Hunter (valid as of 2026-10-04)

```sh
module unload cray-mpich cce
module use /opt/hlrs/testing/modulefiles/rocm-7.14/
module load mpich
```
