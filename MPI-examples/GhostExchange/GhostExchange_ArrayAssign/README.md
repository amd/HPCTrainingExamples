# MPI Ghost Exchange Optimization Examples

## Changes Between Example Versions
This code contains several implementations of the same ghost exchange algorithm at varying stages 
of optimization:
- **Orig**: Shows a CPU-only implementation that uses MPI, and serves as the starting point for further optimizations. It is recommended to start here!
- **Ver1**: Shows an OpenMP target offload implementation that uses the Managed memory model to port the code to GPUs using host allocated memory for MPI communication.
- **Ver2**: Shows the usage and advantages of using `roctx` ranges to get more easily readable profiling output from Omnitrace.
- **UNDER CONSTRUCTION Ver3**: Explores the effects of moving the allocation of communication buffers to the device.
- **Ver4**: Explores heap-allocating communication buffers once on host.
- **Ver5**: Explores unrolling a 2D array to a 1D array.
- **Ver6**: Explores using explicit memory management directives to specify when data movement should happen.
- **UNDER CONSTRUCTION Ver7**: unclear what changes are present here

<details>
<summary><h3>Background Terminology: We're Exchanging <i>Ghosts?</i></h3></summary>
<h4>Problem Decomposition</h4>
In a context where the problem we're trying to solve is spread across many compute resources, 
it is usually inefficient to store the entire data set on every compute node working to solve our problem.
Thus, we "chop up" the problem into small pieces we assign to each node working on our problem.
Typically, this is referred to as a <b>problem decomposition</b>.<br/>
<h4>Ghosts, and Their Halos</h4>
In problem decompositions, we may still need compute nodes to be aware of the work that other nodes 
are currently doing, so we add an extra layer of data, referred to as a <b>halo</b> of <b>ghosts</b>.
This region of extra data can also be referred to as a <b>domain boundary</b>, as it is the <b>boundary</b> 
of the compute node's owned <b>domain</b> of data.
We call it a <b>halo</b> because typically we need to know all the updates happening in the region surrounding a single compute node's data. 
These values are called <b>ghosts</b> because they aren't really there: ghosts represent data another
 compute node controls, and the ghost values are usually set unilaterally through communication 
between compute nodes. 
This ensures each compute node has up-to-date values from the node that owns the underlying data.
These updates can also be called <b>ghost exchanges</b>.
</details>

## Overview of the Ghost Exchange Implementation
The implementations presented in these examples follow the same basic algorithm.
They each implement the same computation, and set up the same ghost exchange, we just change where computation happens, or specifics with data movement or location. 

The code is controlled with the following arguments:
- `-i imax -j jmax`: set the total problem size to `imax*jmax` elements.
- `-x nprocx -y nprocy`: set the number of MPI ranks in the x and y direction, with `nprocx*nprocy` total processes.
- `-h nhalo`: number of halo layers, typically assumed to be 1 for our diagrams.
- `-t (0|1)`: whether time synchronization should be performed.
- `-c (0|1)`: whether corners of the ghost halos should also be communicated.

The computation done on each data element after setup is a blur kernel, that modifies the value of a
given element by averaging the values at a 5-point stencil location centered at the given element:

`xnew[j][i] = (x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i])/5.0`

The communication pattern used is best shown in a diagram that appears in [Parallel and high performance computing, by Robey and Zamora](https://www.manning.com/books/parallel-and-high-performance-computing):
<p>
<img src="ghost_exchange2.png" \>
</p>
In this diagram, a ghost on a process is represented with a dashed outline, while owned data on a process is represented with a solid line. Communication is represented with arrows and colors representing the original data, and the location that data is being communicated and copied to. We see that each process communicates based on the part of the problem it owns: the process that owns the central portion of data must communicate in all four directions, while processes on the corner only have to communicate in two directions.
