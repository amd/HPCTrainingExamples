# hipify the CUDA pingpong code

This code sends data back and forth between the host and device 50 times and calculates the bandwidth. 

Your job is to `hipify` the code, then compile and run it. 
NOTE: The `#include "hip/hip_runtime.h" doesn't always get added when a code is `hipify`-ed, so it might need to be added manually.

To compile and run:
```
$ make
 
$ sbatch submit.sh
```

Recall that the CPU and GPU are connected with PCIe4 (x16), which has a peak bandwidth of 32 GB/s. What percentage of the peak performance do we achieve?
