## Kernel Pragmas

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/kernel_pragmas` from Training Exercises repository

Download the exercises and go to the directory with the kernel pragma examples

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/kernel_pragmas
```

Setup your environment. You should unset the `LIBOMPTARGET_INFO` environment from previous exercise.

```
unset LIBOMPTARGET_INFO
```

```
export CXX=amdclang++
export LIBOMPTARGET_KERNEL_TRACE=1
export OMP_TARGET_OFFLOAD=MANDATORY
export HSA_XNACK=1
```

The base version 1 code is the Unified Shared memory example from the previous exercises

```
mkdir build && cd build
cmake ..
make kernel1
./kernel1
```

`Kernel2 : add num_threads(64)`

`Kernel3 : add num_threads(64) thread_limit(64)`

On your own: Uncomment line in CMakeLists.txt with -faligned-allocation -fnew-alignment=256

Another option is to add the attribute `(std::align_val_t(128) )` to each new line. For example:

```
double *x = new (std::align_val_t(128) ) double[n];
```

