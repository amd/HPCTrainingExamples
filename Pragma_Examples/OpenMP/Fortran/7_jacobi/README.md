# Porting a small app: jacobi

cd 0_jacobi_portyourself
decide if you want to port with or without unified shared memory.
Either for USM
export HSA_XNACK=1
or without for a behaviour with copies
export HSA_XNACK=0

make
run the code for a reference CPU solution.
./jacobi
you can specify a problem size with
./jacobi -m <Problemsize>
otherwise a default of <Problemsize> = 4096 is set.
On the CPU the default problem size may take a minute to get solved, on the GPU it should be solved in a matter of second(s).

port the loops

think about data management or !$omp requires unified_shared_memory.

Example solutions:
For the unified shared memory version:
export HSA_XNACK=1
cd 1_jacobi_usm
make

For the discrete GPU / manual memory management version:
export HSA_XNACK=0
cd 2_jacobi_targetdata
make

Note: In the beta release of the amdflang-new/4.0 compiler HSA_XNACK=0 with a code with !$omp requires unified_shared_memory can be compiled as if no unified_shared_memory is required. This is a behaviour not according to the intended behaviour and will lead to an error message in future releases!

