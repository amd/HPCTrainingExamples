# MPI Ghost Exchange Optimization HIP Examples
This series of examples walk you through several implementations of the same Ghost Exchange
algorithm at varying stages of optimization. Our starting point is the original CPU-only 
implementation found at 
[HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig).
In this series, we use the HIP programming model for offloading compute to the GPU. For 
a good overview of the Ghost Exchange implementation, refer to 
[the documentation here](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign).

## Changes Between Example Versions
Brief descriptions of the various versions of the Ghost Exchange HIP implementation can be found below:
- **Ver1**: Shows a HIP implementation that uses the Managed memory model to port the code to GPUs using 
host allocated memory for work buffers and MPI communication
- **Ver2**: Shows the usage and advantages of using `roctx` ranges to get more easily readable profiling output from Omnitrace
- **Ver4**: Explores heap-allocating communication buffers once on host
- **Ver5**: Explores unrolling a 2D array to a 1D array
- **Ver6**: Explores using explicit memory management directives to reduce data access latency
- **Ver8**: Under Construction, showcases overlap of compute and communication

