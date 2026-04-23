#!/bin/bash
NY=1024 ; NZ=1024 ; NX=256 ; TBSIZE=256; NUMTIMES=1000

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   if [ -z "$HIPCC" ]; then
      export HIPCC=`which hipcc`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

cd ${BUILD_DIR}

git clone --branch v5.0 https://github.com/UoB-HPC/BabelStream.git
cd BabelStream

# Patch BabelStream v5.0 HIP destructor: when MEM=PAGEFAULT the d_a/d_b/d_c
# arrays are malloc()'d but the destructor unconditionally calls hipFree() on
# them, which returns hipErrorInvalidValue and makes the process exit 1
# (prints "Error: invalid argument" after the benchmark results).  Guard the
# hipFree block so PAGEFAULT uses plain free().  Harmless for MEM=DEFAULT.
sed -i '/^HIPStream<T>::~HIPStream()$/,/^}$/{
  s|^  hipFree(d_a);$|#if defined(PAGEFAULT)\
  free(d_a); free(d_b); free(d_c);\
#else\
  hipFree(d_a);|
  /^}$/i\
#endif
}' src/hip/HIPStream.cpp

# -DDEFAULT -- good performance
# -DMANAGED -- poor performance
# -DPAGEFAULT -- good performance
# CMAKE_CXX_COMPILER=hipcc
#TBSIZE = 1024 default -- not seeing much difference
# -DDOT_READ_DWORDS_PER_LANE=4

ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//' | tr -d '[:blank:]'`

# MI300A is an APU with unified host/device memory: DEFAULT mode double-counts
# the allocation (host+device ~= 12.8 GB at this problem size).  Use PAGEFAULT
# (host-only pointers, accessed from the GPU via XNACK/page-fault) to halve
# the footprint and avoid OOM when other GPU tests run concurrently.  Leave
# DEFAULT on discrete GPUs (e.g. MI300X) so we still measure device bandwidth.
MEM_FLAG=""
OFFLOAD_ARCH="${ROCM_GPU}"
export HSA_XNACK=0
if rocminfo 2>/dev/null | grep -q "MI300A"; then
   MEM_FLAG="-DMEM=PAGEFAULT"
   OFFLOAD_ARCH="${ROCM_GPU}:xnack+"
   export HSA_XNACK=1
fi

#hipcc -DTBSIZE=${TBSIZE} -DDOT_READ_DWORDS_PER_LANE=4 \
#   --offload-arch=${ROCM_GPU} -std=c++17 -O3 -DHIP \
#   src/main.cpp src/hip/HIPStream.cpp -o hip-stream -I src/hip -I src

+cmake -Bbuild -H. -DMODEL=hip ${MEM_FLAG} \
+      -DCXX_EXTRA_FLAGS="-D__HIP_PLATFORM_AMD__ --offload-arch=${OFFLOAD_ARCH} -DTBSIZE=${TBSIZE} -DDOT_READ_DWORDS_PER_LANE=4" \
      -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc
cmake --build build

./build/hip-stream --numtimes ${NUMTIMES} --arraysize $((NX*NY*NZ))

# Sample output
#BabelStream
#Version: 5.0
#Implementation: HIP
#Running kernels 1000 times
#Precision: double
#Array size: 2147.5 MB (=2.1 GB)
#Total size: 6442.5 MB (=6.4 GB)
#Using HIP device AMD Instinct MI300A
#Driver: 60240092
#Memory: DEFAULT
#Init: 0.151691 s (=42470.981396 MBytes/sec)
#Read: 0.209183 s (=30798.154971 MBytes/sec)
#Function  MBytes/sec Min (sec) Max    Average
#Copy    3859021.370 0.00111  0.01952  0.00142
#Mul    3789464.147 0.00113  0.00176  0.00137
#Add    3719009.286 0.00173  0.00209  0.00186
#Triad   3787553.768 0.00170  0.02158  0.00187
#Dot    3105050.025 0.00138  0.01149  0.00204

TEMP_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $5}' | echo $(cat $1)`
echo "TEMP: ${TEMP_LIST}"
SCLK_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $10}' | echo $(cat $1)`
echo "SCLK: ${SCLK_LIST}"
MCLK_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $11}' | echo $(cat $1)`
echo "MCLK: ${MCLK_LIST}"
PWR_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $14}' | head -1`
echo "PWR: ${PWR_LIST}"

