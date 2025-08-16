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

Note that in these examples we are assuming for simplicity that `BLOCKSIZE` is a power of two: this requirement can be relaxed but it is not shown here.

## Reduction with No Striding

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
  shared_mem[lid] = 0.0;
  if (gid < size) {
     shared_mem[lid] = input[gid];
  }

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

Remember that threads are executing this code concurrently and each one is modifying locally their entry of `shared_mem`. Hence, to avoid race conditions, we need a synchronization barrier: a thread will continue only when all the other threads in the block have reached the synchronization barrier.

This is the first part of the reduction operation: at the beginning of the for loop, the lower half of the threads in the block start accumulating the values of the input array and store it into their respective entry of the `shared_mem` array. This process is performed recursively with a for loop until the total sum has been accumulated to the first entry of the `shared_mem` array, the one managed by the thread with `lid=0`.

Note that a synchronization barrier is also included as the final step of each iteration of the for loop. This is because the `shared_mem` array has been modified and we need to make sure all threads see the correct version before the next iteration of the for loop.

Here we require the thread with local ID equal to 0 to store the partial sum computed by the threads in the current workgroup on the output array.
Note that the size of the output array is precisely the number of workgroups in the thread grid. This is because the reduction is performed at the workgroup level. At this point, the kernel has computed its execution, but the reduction operation is not yet completed: we need to sum all the partial sums for each workgroup to obtain the final result. This is done again with a reduction operation. In this example, this final reduction is done on the CPU. See `reduction_two_kernel_calls.cpp` for an example of how to call two kernels to perform all the reduction operations on the GPU.


## Reduction with Striding

Consider the following snippet of code from `reduction_striding.cpp`, representing the main kernel in the example:

```
__global__ void get_partial_sums(const double* input, double* output, int size) {
  __shared__ double local_sum[BLOCKSIZE];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  local_sum[threadIdx.x]  = 0.0;
  for (int i = idx; i < size; i += grid_size) {
    local_sum[threadIdx.x]  += input[i];
  }

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

Another way to fix the above example and actually produce correct code is with the use of atomics to perform the reduction, see `reduction_atomic.cpp`:

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

## Reduction with Two Kernel Calls

Consider the kernel in the `reduction_two_kernel_calls.cpp` example:

```
__global__ void reduction_to_array(const double* input, double* output, int size) {
  extern __shared__ double local_sum[];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  local_sum[threadIdx.x] = 0.0;
  for (int i = idx; i < size; i += grid_size) {
     local_sum[threadIdx.x] += input[i];
  }

  // Store local sum in shared memory
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
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

In this example, we will be performing all the reduction operations on the GPU: we will do so by calling the above kernel twice. Notice that in the second for loop, to initialize `s`, we are using `blockDim.x` and not `BLOCKSIZE` as in the previous examples: this is because we will be calling the same kernel `reduction_to_array` twice, but supplying a different thread grid for each invocation. The first invocation proceeds in the same way as in the previous examples, there is a notable difference though which is that now the shared memory is declared as:

```
extern __shared__ double local_sum[];
```

and then the amount to allocate is actually specified through the optional argument at kernel launch:
```
// Compute the reductions
reduction_to_array<<<GRIDSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_in, d_partial_sums, N);
reduction_to_array<<<1, GRIDSIZE, GRIDSIZE*sizeof(double)>>>(d_partial_sums, d_in, GRIDSIZE);
```

The second invocation above is accumulating the partial sums in the first entry of `d_in` that now is supplied as output to the kernel. Notice that the grid that is used for the second kernel invocation is one that has only one block, with size equal to the initial block size. This means that `GRIDSIZE` cannot be larger than 1024, which is the maximum block size. Also notice that because of `blockDim.x/2` in the second for loop in the kernel, we also need `GRIDSIZE` to be a power of 2, in addition to `BLOCKSIZE`, as we have mentioned at the beginning of this README. The requirements of `GRIDSIZE` and `BLOCKSIZE` being powers of two can be relaxed, but such cases are not shown in these examples.

## Reduction with Two Different Kernels and Unrolling

In the previous example, there is an implicit unrolling in each kernel. By unrolling, we mean that each thread adds more than one value from the input array. In this
example, we start off with a simpler technique where in the first kernel we just sum one value per thread. When looking at the source code `reduction_two_kernels_unroll.cpp`, first 
consider it with the `unroll_factor` variable set to one.

```
__global__ void get_partial_sums(const double* input, double* output, int size) {
  extern __shared__ double local_sum[];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  local_sum[threadIdx.x] = 0.0;
  if (idx < size) {
    local_sum[threadIdx.x] += input[idx];
  }

  // Store local sum in shared memory
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
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

We call this kernel with the more usual ceiling function to calculate the number of workgroups. This will than
take the input data array and sum each workgroup. It writes out a `partial_sum` array that is the original input
data size divided by the number of workgroups. 

```
 int nblocks = ceil(N/BLOCKSIZE);
 get_partial_sums<<<nblocks, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_in, d_partial_sums, N);
```

This will in general be a little slower than the `two_kernel_calls` version because there is not enough 
work for each thread and more data needs to be written out. So we add an unrolling factor. First, the calls
to launch the kernel become

```
 int nblocks = ceil(N/BLOCKSIZE/unroll_factor);
 get_partial_sums<<<nblocks, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_in, d_partial_sums, N);
```

Note that all that changes is the calculation of `nblocks`.

```
__global__ void get_partial_sums(const double* input, double* output, int size) {
  extern __shared__ double local_sum[];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  local_sum[threadIdx.x] = 0.0;
  for (int i = idx; i < size; i += grid_size) {
    local_sum[threadIdx.x] += input[i];
  }

  // Store local sum in shared memory
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
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

Note that there is now an extra loop that will effectively cause each thread to sum up to the `unroll_factor` number of values. The unroll factor is
not explicitly sent into the kernel -- it comes in through the new value for the `grid_size`. And we can now see the similarities to the `two_kernel_calls`
version which sends in a fixed number of workgroups. Try varying the unroll factor and see how the performance changes. 

## Reduction with Warp Shuffles

Warp shuffles load adjacent values from threads in a workgroup. The last example shows a version that uses warp shuffles. The benefits
of using warp shuffles is that it reduces shared memory usage (LDS) and also reduces synchronization calls. 


```
make reduction_shfl
./reduction_shfl
```

## Reduction using rocPrim call

Using the rocPrim library for many common operations can greatly simplify programming
while getting good peformance. To see how to use the rocPrim reduction call, see the 
reduction_prim.cpp file. Then build and run it with the following.

```
make reduction_prim
./reduction_prim
```

