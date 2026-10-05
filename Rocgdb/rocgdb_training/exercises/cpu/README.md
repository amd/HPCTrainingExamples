- [Prerequisites](#orgaffc37c)
  - [Start an interactive job](#org838a644)
    - [Hunter](#org4844b62)
  - [Load the required modules](#orgdd6a795)
    - [Hunter (valid as of 2026-10-02)](#orgd34983c)
- [Layout](#orgb2bee13)
- [Building the programs](#org0e290f3)
- [Exercises](#orgc00d433)

These are the hands-on exercises for the "Basics of rocgdb" course (Part I).


<a id="orgaffc37c"></a>

# Prerequisites


<a id="org838a644"></a>

## Start an interactive job

The login node is not for running jobs and on most systems has no GPUs anyway, so for GPU jobs it is not even possible. We therefore start by allocating a short interactive job. The instructions below are not exhaustive: other ways exist, and any preferred options may be used.


<a id="org4844b62"></a>

### Hunter

```sh
# allocate a 10-minute long interactive job
qsub -l select=1:node_type=mi300a -l walltime=0:10:00 -I
```


<a id="orgdd6a795"></a>

## Load the required modules

Before building or running anything, load the toolchain modules for your system.


<a id="orgd34983c"></a>

### Hunter (valid as of 2026-10-02)

```sh
module unload cray-mpich
module unload cce
module use /opt/hlrs/testing/modulefiles/rocm-7.14/
module load mpich
```


<a id="orgb2bee13"></a>

# Layout

-   `common/`: the shared sample programs and their `Makefile` (build once, use everywhere).
-   `NN_name/`: one directory per exercise, numbered in the order the course covers them (`01_running/` is the first). Independent, so any exercise can be skipped.
-   Each exercise directory holds a `README.md` with self-contained instructions; the same text is reproduced in the course PDF's "Hands-on" chapter.


<a id="org0e290f3"></a>

# Building the programs

First build the shared programs once: go into `common/` and follow the `README.md` there. Every exercise then runs rocgdb from its own directory, on `../common/<program>`.


<a id="orgc00d433"></a>

# Exercises

The exercises follow the order of the course material, but are independent, so while it is recommended to do them in numbered order, they can be done in any order and any exercise can also be skipped.
