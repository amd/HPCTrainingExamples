# hipify the CUDA pingpong code

This code sends data back and forth between the host and device 50 times and calculates the bandwidth. 

Your job is to `hipify` the code, then compile and run it. For this exercise, it is recommend to use `hipify-perl` on the CUDA program and redirect the output to a new file titled `pingpong.cpp`. 
NOTE: The `#include "hip/hip_runtime.h" doesn't always get added when a code is `hipify`-ed, so it might need to be added manually.

To compile and run:
```
$ make

$ sbatch -A <account-name> submit.sh
```
where `account-name` is your assigned Frontier username. A job file titled `<name-of-exercise>-%J.out` will be produced, where `%J` is the job id number of your run. To check your program output, simply run:
```
cat <name-of-exercise>-%J.out
```
or open the file directly using `vim`.

Recall that the CPU and GPU are connected with PCIe4 (x16), which has a peak bandwidth of 32 GB/s. What percentage of the peak performance do we achieve?
