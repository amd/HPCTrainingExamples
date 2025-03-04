# HIP Reduction Examples

Clone the repo and get into the relevant directory for this README:

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/HIP/reduction
```

Setup the environment and compile by doing:

```
module load rocm
make
```

## Reduciton with No Striding

Consider the following snippet of code from `reduction_no_striding.cpp`, representing the main kernel in the example:

```
__global__ void get_partial_sums(const double* input, double* output, int size) {
  __shared__ double shared_mem[BLOCKSIZE];

  // Local ID of thread in workgroup (block)
  int lid = threadIdx.x;

  // Global ID of thread in thread grid
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Note: gid has to span all entries of input
  // this means there need to be enough threads
  // on the thread grid to do so
  shared_mem[lid] = input[gid];

  __syncthreads();

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (lid < s) {
      shared_mem[lid] += shared_mem[lid + s];
    }
    __syncthreads();
  }

  if (lid == 0) {
    output[blockIdx.x] = shared_mem[0];
  }
}
```

The kernel receives an array as input and returns another array (not a scalar) as output. This is because the computation will be done on a workgroup fashion, since synchronization of threads is possible only among threads in the same workgroup, hence every workgroup will produce a partial sum.

Note that inside the kernel there is also the allocation of shared memory: the array `shared_mem` is visible to all threads in the workgroup. 

As the name suggest, `lid` is the local ID of the thread within its workgroup, whereas `gid` is the global ID of the thread within the entire thread grid. Here, there is the implicit assumption that there are at least the same number of threads as the entries of the input array input (the size of the grid depends on the size of the input array).

The `shared_mem` array contains the intermediate sums that are computed during the reduction. Each thread in the workgroup has its own intermediate sum, and it is initialized to the entry of the input array that corresponds to the thread’s global ID.

Remember that all threads are executing this code at the same time and each one is modifying locally their entry of `shared_mem`. Hence, to make the changes visible to all the other threads in the workgroup, we need a synchronization barrier. After that call, the array `shared_mem` is the same for all threads in the workgroup. 

This is the actual reduction operation: at the beginning of the for loop, the lower half of the threads in the block start accumulating the values of the input array and store it into their respective entry of the `shared_mem` array. This process is performed recursively with a for loop until the total sum has been accumulated to the first entry of the `shared_mem` array, the one managed by the thread with `lid=0`.

Note that a synchronization barrier is also included as the final step of each iteration of the for loop. This is because the `shared_mem` array has been modified and we need to make sure all threads see its updated version before the next iteration of the for loop.

Here we require the thread with local ID equal to 0 to store the partial sum computed by the threads in the current workgroup on the output array.
Note that the size of the output array is precisely the number of workgroups in the thread grid. This is because the reduction is performed at the workgroup level. At this point, the kernel has computed its execution, but the reduction operation is not yet completed: we need to sum all the partial sums for each workgroup to obtain the final result. 

## Reduction with Striding

Consider the following snippet of code from `reduction_striding.cpp`, representing the main kernel in the example:

```
__global__ void get_partial_sums(const double* input, double* output, int size) {
  __shared__ double local_sum[BLOCKSIZE];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  double sum = 0;
  for (int i = idx; i < size; i += grid_size) {
    sum += input[i];
  }

  // Store local sum in shared memory
  local_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = local_sum[0];
  }
}

```

Note that in this example the number of blocks does not depend on the input array size N because we are using striding.

In this implementation, each thread computes its own partial sum, summing multiple values of the input array, according to the stride size.

Then the `local_sum` array is initialized with each thread's partial sum and a synchronization barrier is called to update the values of this array for each thread. The code then proceeds as the other implementation.
 
## Reduction to Fix

``NOTE``: unlike the two previous examples, this is NOT intended to show how to implement a reduction. 

Consider the following snippet of code from `reduction_to_fix.cpp`, representing the main kernel in the example:

```
__global__ void get_partial_sums_to_fix(const double* input, double* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = gridDim.x * blockDim.x;

  double local_sum = 0;
  for (int i = idx; i < size; i += global_size) {
    local_sum += input[i];
  }

  output[blockIdx.x] = local_sum;
}
```

If you run this example, you will see a failure. 

The reason is that multiple threads are computing local sums with a stride as in the previous example, however multiple threads (namely, those in the same block) write to the same location of the output array, hence not considering all the partial sums that should be considered.

Due to the fact that the input array is all initialized with the same value, a way to fix this example is to multiply the `local_sum` by the workgroup size:

```
output[blockIdx.x] = local_sum * BLOCKSIZE;
```
In this way, we are taking into account all the local sums that are not considered. Note that some unnecessary work is still done and then thrown out.

## Reduction with Atomics

Another way to fix the above example and actually produce correct code is with the use of atomics to perform the reduction, see `reduction_atomics.cpp`:

```
__global__ void atomic_reduction(const double* input, double* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = gridDim.x * blockDim.x;

  double local_sum = 0;
  for (int i = idx; i < size; i += global_size) {
    local_sum += input[i];
  }

  atomicAdd(output,local_sum);
}
```

In the above code, the local sums are accumulated to `output` with an atomic operation: `atomicAdd`. Note that this operation will not do any unsafe atomic operations unless the option `-munsafe-fp-atomics` is provided to the compiler at compile time. Try to replace `atomicAdd` with `unsafeAtomicAdd` which will always implement unsafe atomic operations regardless of what option is supplied to the compiler. How does the exectuion time change? 
