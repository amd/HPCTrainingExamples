
LIBOMPTARGET_DEBUG=<Num>
   Need to build with -DOMPTARGET_DEBUG

More debugging
   -fopenmp-target-debug=<N> at compile time
   LIBOMPARGET_DEVICE_TRL_DEBUG=<N> at run time

   0x01: Enable debugging assertions in the device
   0x2: Enable OpenMP runtime function traces in the device
   0x4: Enable diagnosing common problems during offloading 

LIBOMPTARGET_PROFILE=<Filename>
   Need to set CMake option OPENMP_ENABLE_LIBOMP_PROFILING=ON
   View with chrome::/tracing

LIBOMPTARGET_INFO=<Num>
   0x01: Print all data arguments upon entering an OpenMP device kernel
   0x02: Indicate when a mapped address already exists in the device mapping table
   0x04: Dump the contents of the device pointer map at kernel exit
   0x08: Indicate when an entry is changed in the device mapping table
   0x10: Print OpenMP kernel information from device plugins
   0x20: Indicate when data is copied to and from the device
   -1: Enable all
   Example LIBOMPTARGET_INFO=$((0x1 | 0x10)) 

OMP_TARGET_OFFLOAD MANDATORY

The default memory alignment obtained with `new` 16 bytes. Such alignment is not optimal 
for computing on GPUs. 

C++ has methods to specify memory alignment using default parameters set at 
compilation time (-faligned-allocation -fnew-alignment=64) at run time as shown 
in the example. Use of system memory allocators uch as posix_memalign is also an 
alternative.

Y =  new (std::align_val_t(64)) double[N];

USM needs to be run with HSA_XNACK=1 and xnack-any or xnack-on compiler flags (no flag about xnack is xnack-any)

