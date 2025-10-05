
# GPU Aware MPI

README.md from `HPCTrainingExamples/MPI-examples` from the Training Examples repository.

## Point-to-point and collective

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

Allocate at least two GPUs and set up your environment

```
module load openmpi rocm
export OMPI_CXX=hipcc
```

Find the code and compile
```
cd HPCTrainingExamples/MPI-examples
mpicxx -o ./pt2pt ./pt2pt.cpp
```
Set the environment variable and run the code
```
mpirun -n 2 -mca pml ucx ./pt2pt
```

## OSU Benchmark

Get the OSU micro-benchmark tarball and extract it
```
mkdir OMB
cd OMB
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz
tar -xvf osu-micro-benchmarks-7.3.tar.gz
```

Create a build directory and cd to osu-micro-benchmarks-7.3
```
mkdir build
cd osu-micro-benchmarks-7.3
module load rocm openmpi
```

Build and install OSU micro-benchmarks
```
./configure --prefix=`pwd`/../build/ \
                CC=`which mpicc` \
                CXX=`which mpicxx` \
                CPPFLAGS=-D__HIP_PLATFORM_AMD__=1 \
                --enable-rocm \
                --with-rocm=${ROCM_PATH}
make -j12
make install
```
If you get the error "cannot include hip/hip_runtime_api.h", grep for __HIP_PLATFORM_HCC__ and replace it with __HIP_PLATFORM_AMD__ in configure.ac and configure files.

Check if osu microbenchmark is actually built
```
ls -l ../build/libexec/osu-micro-benchmarks/mpi/

```
if you see files collective, one-sided, pt2pt, and startup, your build is successful.

Allocate 2 GPUs, and make those visible
```
export HIP_VISIBLE_DEVICES=0,1
```

Make sure GPU-Aware communication is enabled and run the benchmark
```
mpirun -n 2 -mca pml ucx ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw \
				-m $((16*1024*1024)) D D
```


Notes:
- Try different pairs of GPUs.
- Run the command "rocm-smi --showtopo" to see the link type between the pairs of GPUs. 
- How does the bandwidth vary for xGMI connected GPUs vs PCIE connected GPUs?

## Ghost Exchange example

The Ghost Exchange example is a simplified instance of what we believe a real scientific application code that uses MPI might look like. There are OpenMP and HIP versions of this example in 2D, each of which has multiple implementations tackling progressive code improvements. For detailed instructions, see the dedicated [directory](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange). Low detail, quick start instructions are reported here for people that want to experiment quickly and are OK with filling in the blanks on their own.

For what follows, we focus on the 2D [OpenMP version set](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign), which begins with a CPU only version that can be compiled and run as below:
```
module load amdclang openmpi
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig
mkdir build && cd build
cmake ..
make -j
mpirun -n 8 --mca pml ucx ./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```
We can improve this performance by using process placement so that we are using all the memory
channels.

On MI210 nodes, we have 2 NUMA per node. So we can assign 4 ranks per NUMA when running with 8 ranks:

```
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:4:numa --report-bindings ./GhostExchange \
				    -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

On MI300A node, we have 4 NUMA per node. So we can assign 2 ranks per NUMA when running with 8 ranks:

```
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:2:numa --report-bindings ./GhostExchange \
				    -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```
Let's consider the OpenMP version from now on: in Ver1, the original CPU implementation is ported to GPU using OpenMP and unified shared memory (or single memory space when running on MI300A). This is enabled with `export HSA_XNACK=1` as shown below. For MI210 we have:

```
export HSA_XNACK=1
cd ../Ver1
mkdir build && cd build
cmake ..
make -j
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:4:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \ 
					./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

Alternatively, on MI300A, we can run with:

```
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:2:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
				    ./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

The MPI communication buffers up to this point were allocated on the CPU, we can allocate them on the GPU and save memory on the CPU, while at the same time leveraging GPU-aware MPI, as shown in Ver3. For MI210:

```
export HSA_XNACK=1
cd ../Ver3
mkdir build && cd build
cmake ..
make -j
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:4:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \ 
					./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

Alternatively, on MI300A, we can run with:

```
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:2:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
				    ./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

Memory allocations can be expensive for the GPU. This next version just allocates the MPI buffers dynamically once
in the main routine.

```
export HSA_XNACK=1
cd ../Ver4
mkdir build && cd build
cmake ..
make -j
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:4:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
					./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

Alternatively, on MI300A, we can run with:

```
mpirun -n 8 --mca pml ucx --bind-to core --map-by ppr:2:numa -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
				   ./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 1000
```

Two more versions are available in the dedicated [directory](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign), which are not discussed here.

## RCCL Test

To run RCCL test, follow these steps:

```
module load rocm
module load openmpi
git clone https://github.com/ROCm/rccl-tests.git
cd rccl-tests/
make MPI=1 MPI_HOME=$MPI_PATH HIP_HOME=$ROCM_PATH
```

where above `MPI_PATH` and `ROCM_PATH` are set by loading the `openmpi` and `rocm` modules respectively, according to the installation of OpenMPI and ROCm provided in our [HPCTrainingDock](https://github.com/amd/HPCTrainingDock) repo.

After successful build, you should be able to see the executables in `./build` directory. You can run the collectives with:

```
./build/all_reduce_perf -b 4M -e 128M -f 2 -g 4
```

The above command will run for 4M (`-b`) to 128M (`-e`) messages, with a  multiplication factor between sizes equal to 2 (`-f`), and using 4 GPUs (`-g`).
