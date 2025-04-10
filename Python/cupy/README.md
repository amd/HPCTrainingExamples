## CuPy Examples

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

### Simple introduction example to CuPy for AMD GPUs

To run this example, 

```
module load cupy
python cupy_array_sum.py
```

The output should look like the following:

```
CuPy Array: [1 2 3 4 5]
Squared CuPy Array: [ 1  4  9 16 25]
NumPy Array: [5 6 7 8 9]
CuPy Array from NumPy: [5 6 7 8 9]
Addition Result on GPU: [ 6  8 10 12 14]
Result on CPU: [ 6  8 10 12 14]
```

What is actually happening here? What is on the GPU and what is on the CPU?
Let's try and see more.

```
export AMD_LOG_LEVEL=1
python cupy_array_sum.py
```

Now our output is:

```
:1:hip_memory.cpp           :3721: 1083559518128d us:  Cannot get amd_mem_obj for ptr: 0x46b8a5f0
CuPy Array: [1 2 3 4 5]
Squared CuPy Array: [ 1  4  9 16 25]
NumPy Array: [5 6 7 8 9]
:1:hip_memory.cpp           :3721: 1083560370823d us:  Cannot get amd_mem_obj for ptr: 0x483ba890
CuPy Array from NumPy: [5 6 7 8 9]
Addition Result on GPU: [ 6  8 10 12 14]
Result on CPU: [ 6  8 10 12 14]
```

The warning is from the AMD logging functions and doesn't impact the run. Now let's increase the log level for the run.

```
export AMD_LOG_LEVEL=3
python cupy_array_sum.py
```

Now we see lots of output that shows the hip calls and the operations on the GPU.

```
....
hipMemcpyAsync ( 0x559ea98f65f0, 0x7f4556800000, 40, hipMemcpyDeviceToHost, stream:<null> )
Signal = (0x7f4d5efff280), Translated start/end = 1083534945452078 / 1083534945453358, Elapsed = 1280 ns, ticks start/end = 27091222405615 / 27091222405647, Ticks elapsed = 32
Host active wait for Signal = (0x7f4d5efff200) for -1 ns
Set Handler: handle(0x7f4d5efff180), timestamp(0x559eaabead90)
Host active wait for Signal = (0x7f4d5efff180) for -1 ns
hipMemcpyAsync: Returned hipSuccess : : duration: 5948d us
hipStreamSynchronize ( stream:<null> )
Handler: value(0), timestamp(0x559eaa7e7350), handle(0x7f4d5efff180)
hipStreamSynchronize: Returned hipSuccess :
hipSetDevice ( 0 )
hipSetDevice: Returned hipSuccess :
CuPy Array: [1 2 3 4 5]
....
```
