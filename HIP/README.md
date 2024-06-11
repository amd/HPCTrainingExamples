# Introduction to HIP Exercises

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

`git clone https://github.com/amd/HPCTrainingExamples.git`

For the first interactive example, get an slurm interactive session

`salloc -N 1 -p LocalQ --gpus=1 -t 10:00`

## Basic examples

`cd HPCTrainingExamples/HIP/vectorAdd `

Examine files here - README, Makefile, CMakeLists.txt and vectoradd.hip. Notice that Makefile requires `ROCM_PATH` to be set. Check with module show rocm or echo `$ROCM_PATH` Also, the Makefile builds and runs the code. We'll do the steps separately. Check also the HIPFLAGS in the Makefile. There is also a CMakeLists.txt file to use for a cmake build.

For the portable Makefile system
```bash
module load rocm

make vectoradd
./vectoradd
```

Pro tip for Makefile builds. Run `make clean` before `make` to be sure nothing is left over from a previous build.

This example also runs with the cmake system

```bash
module load rocm

mkdir build && cd build
cmake ..
make
./vectoradd
```

Pro tip for cmake builds. To rebuild after changing CMake options or using a different compiler, either

* Remove the CMakeCache.txt, or
* clean out all files from the ./build directory

We can use a SLURM submission script, let's call it `hip_batch.sh`. There is a sample script for some systems in the example directory.

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -p LocalQ
#SBATCH --gpus=1
#SBATCH -t 10:00

module load rocm
cd $HOME/HPCTrainingExamples/HIP/vectorAdd 

make vectoradd
./vectoradd
```

Submit the script
`sbatch hip_batch.sh`

Check for output in `slurm-<job-id>.out` or error in `slurm-<job-id>.err`

To use the cmake option in the batch file, change the build to

```bash
mkdir build && cd build
cmake ..
make
./vectoradd
```

Now let's try the hip-stream example. This example is from the original McCalpin code as ported to CUDA by Nvidia. This version has been ported to use HIP.

```bash
module load rocm
cd $HOME/HPCTrainingExamples/HIP/hip-stream
make
./stream
```
Note that it builds with the hipcc compiler. You should get a report of the Copy, Scale, Add, and Triad cases.

On your own:

1. Check out the saxpy example in `HPCTrainingExamples/HIP`
2. Write your own kernel and run it
3. Test the code on an Nvidia system -- Add `HIPCC=nvcc` before the make command or `-DCMAKE_GPU_RUNTIME=CUDA` to the cmake command. (See README file)

## More advanced HIP makefile

The jacobi example has a more complex build that incorporates MPI. The original Makefile has not been modified, but a CMakeLists.txt has been added to demonstrate a portable cmake build. From an interactive session, build the example.


```bash
cd $HOME/HPCTrainingExamples/HIP/jacobi

module load rocm
module load openmpi

mkdir build && cd build
cmake ..
make
```

Since we will be running on two MPI ranks, you will need to alloc 2 GPUs for a quick run. Exit your current allocation with `exit` and then get the two GPUs. Keep the requested time short to avoid tying up the GPUs so others can run the examples. The requested time shown is in the format hours:minutes:seconds so it is for one minute.

```bash 
salloc -p LocalQ --gpus=2 -n 2 -t 00:01:00
module load rocm openmpi
mpirun -n 2 ./Jacobi_hip -g 2
```
