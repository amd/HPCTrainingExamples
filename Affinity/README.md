# Affinity Exercises

In this set of exercises, we will take two example applications in tandem to show the effect of setting proper affinity to CPU cores and GPUs. Using the `hello_jobstep` example, we will see how affinity settings could be verified programmatically. Using the `ghost exchange` example, we will examine how setting affinity properly can help improve performance. We will examine different scenarios incrementally starting from not setting any affinity at all, to setting both CPU core and GPU device affinities. Please note that the `hello_jobstep` application is built with OpenMP&reg; support while the `Ghost Exchange Ver1` example does not have OpenMP support.

## Build examples

Follow the instructions in the sections below for cloning and building the two examples.

### Build `hello_jobstep` example

Clone the `hello_jobstep` code and build it as shown below:

```
cd ~/git
git clone https://code.ornl.gov/olcf/hello_jobstep.git
cd hello_jobstep
cat >> Makefile.new
SOURCES = hello_jobstep.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = hello_jobstep

CXX=g++
CXXFLAGS = -fopenmp -I/opt/ompi-5.0.3/include -I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__
LDFLAGS = -L${ROCM_PATH}/lib -lhsa-runtime64 -lamdhip64 -L/opt/ompi-5.0.3/lib -lmpi

all: ${EXECUTABLE}

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
Ctrl+C
make
```

### Build `Ghost Exchange` (Ver1) example

To build the `Ghost Exchange` code, we need ROCm to be loaded in your environment and the `ROCM_PATH` environment variable be set up with the path to your ROCm installation. We will also need a build of OpenMPI (say ver 5.0.3) that uses UCX (say ver 1.16.x) built with ROCm. The following instructions assume that ROCM_PATH is set up, and that OpenMPI is installed at `/opt/ompi-5.0.3`. 

[Ver1 of the `Ghost Exchange` example](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1) uses offloading to GPU using the HIP programming model and a managed memory model. Using the managed memory model, the memory buffers are still initially allocated on host, but the OS will manage page migration and data movement across the PCIe link between the host and device. For this to happen when the `Ghost Exchange` example is run, we need to set up an environment variable, `HSA_XNACK=1`. 

```
cd ~/git
git clone git@github.com:amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1

export PATH=/opt/ompi-5.0.3/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/ompi-5.0.3/lib:$LD_LIBRARY_PATH
export HSA_XNACK=1

mkdir build; cd build;
cmake -D CMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++ -D CMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang -DCMAKE_PREFIX_PATH=${ROCM_PATH}/lib/cmake/hip ..
make
```

## System topology

To understand how to set up affinity, we need to know the topology of the system we have. Two commands help us here. `lscpu` informs us the CPU hardware thread (HWT) count and how they are configured into NUMA domains. `rocm-smi --showtoponuma` shows the NUMA configuration for the GPU devices on the system. As an example, here are the details from each command on the node we are using for this tutorial.

The output of `lscpu` below shows that hardware threads 0-23 and 96-119 belong to NUMA domain 0, 24-47 and 120-143 belong to NUMA domain 1, and so on.

```
$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              192
On-line CPU(s) list: 0-191
Thread(s) per core:  2
Core(s) per socket:  24
Socket(s):           4
NUMA node(s):        4
Vendor ID:           AuthenticAMD
CPU family:          25
Model:               144
Model name:          AMD Instinct MI300A Accelerator
Stepping:            1
CPU MHz:             3700.000
CPU max MHz:         3700.0000
CPU min MHz:         1500.0000
BogoMIPS:            7399.66
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            32768K
NUMA node0 CPU(s):   0-23,96-119
NUMA node1 CPU(s):   24-47,120-143
NUMA node2 CPU(s):   48-71,144-167
NUMA node3 CPU(s):   72-95,168-191
```

And the output of `rocm-smi --showtoponuma` shows that GPU device 0 belongs to NUMA domain 0, GPU 1 belongs to NUMA domain 1, etc. With this knowledge, we believe that it would be best if a process using GPU 0 is also pinned to run on a hardware thread in the range 0-23,96-119. The proximity of CPU cores to GPUs is determined by their association with a common NUMA domain.

```
============================ ROCm System Management Interface ============================
======================================= Numa Nodes =======================================
GPU[0]		: (Topology) Numa Node: 0
GPU[0]		: (Topology) Numa Affinity: 0
GPU[1]		: (Topology) Numa Node: 1
GPU[1]		: (Topology) Numa Affinity: 1
GPU[2]		: (Topology) Numa Node: 2
GPU[2]		: (Topology) Numa Affinity: 2
GPU[3]		: (Topology) Numa Node: 3
GPU[3]		: (Topology) Numa Affinity: 3
================================== End of ROCm SMI Log ===================================
```


## Run 1 Rank / GPU

Let us examine the case where we run only 1 MPI process per GPU device. In each case shown below, we will run both examples using the same settings and observe the output from each one.

### Run without affinity settings

In this case, we do not set up any affinity, so we see that each process landed in some HWT. In this case, we see HWTs 001, 121, 144, and 072. We also see all GPUs were available to each process in the `RT_GPU_ID` list for each one, and there was no assigned GPU for each process because `GPU_ID = N/A`.

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 mpirun -np 4 --mca pml ucx --mca coll ^hcoll ./hello_jobstep
MPI 002 - OMP 000 - HWT 144 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 001 - OMP 000 - HWT 121 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 003 - OMP 000 - HWT 072 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 000 - OMP 000 - HWT 001 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
```

The output for the `Ghost Exchange` example shows the same thing, and we see that the elapsed time for this run without any affinity settings is 3.55 seconds.

```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ mpirun -np 4 --mca pml ucx --mca coll ^hcoll ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100

MPI 003 - HWT 072 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A​
MPI 002 - HWT 145 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A​
MPI 000 - HWT 003 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A​
MPI 001 - HWT 024 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A​
​
GhostExchange_ArrayAssign_HIP Timing is stencil 2.443530 boundary condition 0.022016 ghost cell 0.032631 total 3.552828
```

### Run with GPU affinity only

Next, we will set up GPU affinity by setting assigning a unique GPU device for each process via the environment variable, `ROCR_VISIBLE_DEVICES`. We use a script at `~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/set_gpu_device_mi300a.sh` to achieve this. Now, we see that process 0 ran on GPU device 0 (see `GPU_ID` field), and so on.

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 mpirun -np 4 --mca pml ucx --mca coll ^hcoll ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/set_gpu_device_mi300a.sh ./hello_jobstep
MPI 003 - OMP 000 - HWT 081 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
MPI 002 - OMP 000 - HWT 145 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 000 - OMP 000 - HWT 096 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 001 - OMP 000 - HWT 025 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
```

We can observe the GPU affinity set up correctly in the `Ghost Exchange` example too. In this case, we see that the application ran almost 3x faster with an elapsed time of 1.2 seconds.


```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ mpirun -np 4 --mca pml ucx --mca coll ^hcoll ../../set_gpu_device_mi300a.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100​
​
MPI 000 - HWT 097 - RT_GPU_ID 0 - GPU_ID 0​
MPI 003 - HWT 089 - RT_GPU_ID 0 - GPU_ID 3​
MPI 001 - HWT 024 - RT_GPU_ID 0 - GPU_ID 1​
MPI 002 - HWT 048 - RT_GPU_ID 0 - GPU_ID 2​
​
GhostExchange_ArrayAssign_HIP Timing is stencil 0.629131 boundary condition 0.007776 ghost cell 0.074072 total 1.208454
```

### Run with CPU affinity only

Now, we will attempt to set CPU affinity only. 

For the `hello_jobstep` example that supports OpenMP, we will use OpenMP environment variables, `OMP_PLACES` and `OMP_PROC_BIND` to specify placement and binding for each process. Now, we see that when using `OMP_PLACES=threads` and `OMP_PROC_BIND=close`, processes 0, 1, 2, and 3 ran on HWTs 0, 24, 48, and 72 respectively. Please note that you may see different behavior on your system, and hence exploring the OpenMP man page for the various settings for these environment variables may become unavoidable.

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 OMP_PLACES=threads OMP_PROC_BIND=close mpirun -np 4 --mca pml ucx --mca coll ^hcoll ./hello_jobstep
MPI 002 - OMP 000 - HWT 048 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 003 - OMP 000 - HWT 072 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 001 - OMP 000 - HWT 024 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
MPI 000 - OMP 000 - HWT 000 - Node <host> - RT_GPU_ID 0,1,2,3 - GPU_ID N/A - Bus_ID 02,02,02,02
```

In the `Ghost Exchange` case, we need to use a different pinning mechanism for CPU threads as it is not built with OpenMP support. We resort to using OpenMPI binding options as shown below. And we see that setting CPU affinity alone is not sufficient.

```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by ppr:1:socket ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100

MPI 000 - HWT 000 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A
MPI 001 - HWT 024 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A
MPI 003 - HWT 072 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A
MPI 002 - HWT 048 - RT_GPU_ID 0,1,2,3 - GPU_ID N/A

GhostExchange_ArrayAssign_HIP Timing is stencil 2.803742 boundary condition 0.016902 ghost cell 0.052982 total 3.907620
```

### Run with CPU and GPU affinity

Let us set both CPU and GPU affinity now.

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 OMP_PLACES=threads OMP_PROC_BIND=close mpirun -np 4 --mca pml ucx --mca coll ^hcoll ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/set_gpu_device_mi300a.sh ./hello_jobstep
MPI 002 - OMP 000 - HWT 048 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 000 - OMP 000 - HWT 000 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 001 - OMP 000 - HWT 024 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 003 - OMP 000 - HWT 072 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
```

And for Ghost exchange, we use a combination of mpirun binding options and the GPU affinity script. This improves the performance back to what we saw when we pinned the GPU device only. This shows that GPU affinity is very important for this application's performance.

```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by ppr:1:socket ../../set_gpu_device_mi300a.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100

MPI 000 - HWT 096 - RT_GPU_ID 0 - GPU_ID 0
MPI 002 - HWT 049 - RT_GPU_ID 0 - GPU_ID 2
MPI 001 - HWT 121 - RT_GPU_ID 0 - GPU_ID 1
MPI 003 - HWT 083 - RT_GPU_ID 0 - GPU_ID 3

GhostExchange_ArrayAssign_HIP Timing is stencil 0.651226 boundary condition 0.007590 ghost cell 0.080684 total 1.235013
```

## Oversubscribing processes on GPU devices

Oversubscription of a GPU with multiple processes requires a more complex script that helps set up affinity such that neighboring ranks are closely packed on GPU devices (i.e., ranks 0 and 1 on GPU 0, ranks 2 and 3 on GPU 1, etc.) and use different cores on the NUMA domain that is closest to the selected GPU. The script shown below works perfectly for the `hello_jobstep` example as using `GOMP_CPU_AFFINITY` environment variable for setting CPU affinity requires an OpenMP based application.

```
#!/bin/bash

export global_rank=${OMPI_COMM_WORLD_RANK}
export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}
export ranks_per_node=${OMPI_COMM_WORLD_LOCAL_SIZE}

if [ -z "${NUM_CPUS}" ]; then
    let NUM_CPUS=96
fi

if [ -z "${RANK_STRIDE}" ]; then
    let RANK_STRIDE=$(( ${NUM_CPUS}/${ranks_per_node} ))
fi

if [ -z "${OMP_STRIDE}" ]; then
    let OMP_STRIDE=1
fi

if [ -z "${NUM_GPUS}" ]; then
    let NUM_GPUS=4
fi

if [ -z "${GPU_START}" ]; then
    let GPU_START=0
fi

if [ -z "${GPU_STRIDE}" ]; then
    let GPU_STRIDE=1
fi

cpu_list=($(seq 0 95))
let cpus_per_gpu=${NUM_CPUS}/${NUM_GPUS}
let cpu_start_index=$(( ($RANK_STRIDE*${local_rank})+${GPU_START}*$cpus_per_gpu ))
let cpu_start=${cpu_list[$cpu_start_index]}
let cpu_stop=$(($cpu_start+$OMP_NUM_THREADS*$OMP_STRIDE-1))

gpu_list=(0 1 2 3)
let ranks_per_gpu=$(((${ranks_per_node}+${NUM_GPUS}-1)/${NUM_GPUS}))
let my_gpu_index=$(($local_rank*$GPU_STRIDE/$ranks_per_gpu))+${GPU_START}
let my_gpu=${gpu_list[${my_gpu_index}]}

export GOMP_CPU_AFFINITY=$cpu_start-$cpu_stop:$OMP_STRIDE
export ROCR_VISIBLE_DEVICES=$my_gpu

"$@"
```

### Run 2 ranks/GPU with CPU and GPU affinity

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 mpirun -np 8 --mca pml ucx --mca coll ^hcoll ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/set_cpu_gpu_mi300a.sh ./hello_jobstep | sort
MPI 000 - OMP 000 - HWT 000 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 001 - OMP 000 - HWT 012 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 002 - OMP 000 - HWT 024 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 003 - OMP 000 - HWT 036 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 004 - OMP 000 - HWT 048 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 005 - OMP 000 - HWT 060 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 006 - OMP 000 - HWT 072 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
MPI 007 - OMP 000 - HWT 084 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
```

For `Ghost Exchange`, we will use a combination of OpenMPI binding options and the packing script to set up GPU affinity:

```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ OMP_NUM_THREADS=1 mpirun -np 8 --mca pml ucx --mca coll ^hcoll --map-by ppr:2:socket ../../set_cpu_gpu_mi300a.sh ./GhostExchange -x 4  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100

MPI 003 - HWT 136 - RT_GPU_ID 0 - GPU_ID 1
MPI 004 - HWT 145 - RT_GPU_ID 0 - GPU_ID 2
MPI 005 - HWT 159 - RT_GPU_ID 0 - GPU_ID 2
MPI 002 - HWT 122 - RT_GPU_ID 0 - GPU_ID 1
MPI 006 - HWT 080 - RT_GPU_ID 0 - GPU_ID 3
MPI 007 - HWT 168 - RT_GPU_ID 0 - GPU_ID 3
MPI 001 - HWT 001 - RT_GPU_ID 0 - GPU_ID 0
MPI 000 - HWT 016 - RT_GPU_ID 0 - GPU_ID 0
GhostExchange_ArrayAssign_HIP Timing is stencil 0.467487 boundary condition 0.005841 ghost cell 0.212834 total 1.195145
```

### Run 4 ranks per GPU with CPU and GPU affinity:

For `hello_jobstep`:

```
$ cd ~/git/hello_jobstep
$ OMP_NUM_THREADS=1 mpirun -np 16 --mca pml ucx --mca coll ^hcoll ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/set_cpu_gpu_mi300a.sh ~/git/hello_jobstep/hello_jobstep | sort
MPI 000 - OMP 000 - HWT 000 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 001 - OMP 000 - HWT 006 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 002 - OMP 000 - HWT 012 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 003 - OMP 000 - HWT 018 - Node <host> - RT_GPU_ID 0 - GPU_ID 0 - Bus_ID 02
MPI 004 - OMP 000 - HWT 024 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 005 - OMP 000 - HWT 030 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 006 - OMP 000 - HWT 036 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 007 - OMP 000 - HWT 042 - Node <host> - RT_GPU_ID 0 - GPU_ID 1 - Bus_ID 02
MPI 008 - OMP 000 - HWT 048 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 009 - OMP 000 - HWT 054 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 010 - OMP 000 - HWT 060 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 011 - OMP 000 - HWT 066 - Node <host> - RT_GPU_ID 0 - GPU_ID 2 - Bus_ID 02
MPI 012 - OMP 000 - HWT 072 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
MPI 013 - OMP 000 - HWT 078 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
MPI 014 - OMP 000 - HWT 084 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
MPI 015 - OMP 000 - HWT 090 - Node <host> - RT_GPU_ID 0 - GPU_ID 3 - Bus_ID 02
```

And for `Ghost Exchange`:

```
$ cd ~/git/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/HIP/Ver1/build
$ OMP_NUM_THREADS=1 mpirun -np 16 --mca pml ucx --mca coll ^hcoll --map-by ppr:4:socket ../../set_cpu_gpu_mi300a.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I 100

MPI 009 - HWT 057 - RT_GPU_ID 0 - GPU_ID 2
MPI 010 - HWT 048 - RT_GPU_ID 0 - GPU_ID 2
MPI 001 - HWT 010 - RT_GPU_ID 0 - GPU_ID 0
MPI 002 - HWT 011 - RT_GPU_ID 0 - GPU_ID 0
MPI 012 - HWT 184 - RT_GPU_ID 0 - GPU_ID 3
MPI 008 - HWT 071 - RT_GPU_ID 0 - GPU_ID 2
MPI 011 - HWT 068 - RT_GPU_ID 0 - GPU_ID 2
MPI 013 - HWT 074 - RT_GPU_ID 0 - GPU_ID 3
MPI 014 - HWT 177 - RT_GPU_ID 0 - GPU_ID 3
MPI 015 - HWT 168 - RT_GPU_ID 0 - GPU_ID 3
MPI 000 - HWT 023 - RT_GPU_ID 0 - GPU_ID 0
MPI 004 - HWT 126 - RT_GPU_ID 0 - GPU_ID 1
MPI 007 - HWT 025 - RT_GPU_ID 0 - GPU_ID 1
MPI 003 - HWT 099 - RT_GPU_ID 0 - GPU_ID 0
MPI 006 - HWT 041 - RT_GPU_ID 0 - GPU_ID 1
MPI 005 - HWT 038 - RT_GPU_ID 0 - GPU_ID 1
GhostExchange_ArrayAssign_HIP Timing is stencil 0.561575 boundary condition 0.028996 ghost cell 0.059794 total 1.095871
```

Hope this tutorial helped you learn some useful tricks to set affinity properly. Happy pinning!
