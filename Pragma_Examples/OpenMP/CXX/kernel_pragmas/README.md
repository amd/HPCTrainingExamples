Environment variables:
  `CXX=<C++ Compiler>`
      Example: `CXX=amdclang++`
  `LIBOMPTARGET_KERNEL_TRACE=[1,2]`
  `OMP_TARGET_OFFLOAD=MANDATORY`
  `HSA_XNACK=1

Kernel1 : Base kernel with unified shared memory

   set HSA_XNACK=1 at runtime
   set LIBOMPTARGET_KERNEL_TRACE=1

Kernel2 : add num_threads(64)

Kernel3 : add num_threads(64) thread_limit(64)

On your own: Uncomment line in CMakeLists.txt with -faligned-allocation -fnew-alignment=256

Sample kernel optimization clauses
   num_threads(64) num_teams(480) thread_limit(64)
