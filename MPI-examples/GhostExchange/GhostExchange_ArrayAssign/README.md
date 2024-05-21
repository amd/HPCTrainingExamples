# MPI Ghost Exchange Optimization Examples

<details>
<summary><h3>Background Terminology: We're Exchanging <i>Ghosts?</i></h3></summary>
In a context where the problem we're trying to solve is spread across many compute resources, 
it is usually inefficient to store the entire data set on every compute node working to solve our problem.
Thus, we "chop up" the problem into small pieces we assign to each node working on our problem.
Typically, this is referred to as a <b>problem decomposition</b>.
In problem decompositions, we may still need compute nodes to be aware of the work that other nodes 
are currently doing, so we add an extra layer of data, referred to as a "halo" of "ghosts".
This region of extra data can also be referred to as a "domain boundary", as it is the "boundary" 
of the compute node's owned "domain" of data.
We call it a <b>halo</b> because typically we need to know all the updates happening in the region surrounding a single compute node's data. 
These values are called <b>ghosts</b> because they aren't really there: ghosts represent data another
 compute node controls, and the ghost values are usually set unilaterally through communication 
between compute nodes. 
This ensures each compute node has up-to-date values from the node that owns the underlying data.
These updates can also be called "ghost exchanges".
</details>

# Overview of the Ghost Exchange Implementation
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
Here we see that each process is assigned a portion of the problem, and when communicating factors 
its portion of the problem's locality: the MPI rank corresponding to the bottom left corner only communicates up and to the right, where as the rank that has the center of the problem must communicate in all directions to provide comprehensive updates.

# Changes Between Example Versions
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


