- [Prerequisites](#org9e067a8)
  - [Start an interactive job](#org30ce16b)
    - [Hunter](#org5eaa50b)
  - [Load the required modules](#org093874b)
    - [Hunter (valid as of 2026-10-02)](#org5936725)
- [Layout](#org5f9bb74)
- [Building the programs](#org3598b49)
- [Exercises](#org2abe953)

These are the hands-on exercises for the "Basics of rocgdb" course (Part I).


<a id="org9e067a8"></a>

# Prerequisites


<a id="org30ce16b"></a>

## Start an interactive job

The login node is not for running jobs and on most systems has no GPUs anyway, so for GPU jobs it isn't even possible. So, you should start by allocating yourself a short interactive job. There are instructions below for how to do this on various systems. These are not exhaustive: there are other ways, too, you may use your favourite options, too


<a id="org5eaa50b"></a>

### Hunter

```sh
# allocate a 10-minute long interactive job
qsub -l select=1:node_type=mi300a -l walltime=0:10:00 -I
```


<a id="org093874b"></a>

## Load the required modules

Before building or running anything, load the toolchain modules for your system.


<a id="org5936725"></a>

### Hunter (valid as of 2026-10-02)

```sh
module unload cray-mpich
module unload cce
module use /opt/hlrs/testing/modulefiles/rocm-7.14/
module load mpich
```


<a id="org5f9bb74"></a>

# Layout

-   `common/` - the shared sample programs and their `Makefile` (build once, use everywhere).
-   `NN_name/` - one directory per exercise, numbered in the order the course covers them (`01_running/` is the first). Independent, so any exercise can be skipped.
-   Each exercise directory holds a `README` with self-contained instructions; the same text is reproduced in the course PDF's "Hands-on" chapter.


<a id="org3598b49"></a>

# Building the programs

First build the shared programs once: go into `common/` and follow the `README` there. Every exercise then runs rocgdb from its own directory, on `../common/<program>`.


<a id="org2abe953"></a>

# Exercises

The exercises follow the order of the course material, but are independent, so while it is recommended to do them in numbered order, they can be done in any order and any exercise can also be skipped.
