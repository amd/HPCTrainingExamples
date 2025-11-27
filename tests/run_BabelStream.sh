#!/bin/bash
NY=1024 ; NZ=1024 ; NX=256 ; TBSIZE=256; NUMTIMES=1000
BABELSTREAM_ROOT=${PWD}/BabelStream
rm -rf ${BABELSTREAM_ROOT}
if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
git clone https://github.com/UoB-HPC/BabelStream.git ${BABELSTREAM_ROOT}
cd ${BABELSTREAM_ROOT}
# -DDEFAULT -- good performance
# -DMANAGED -- poor performance
# -DPAGEFAULT -- good performance
# CMAKE_CXX_COMPILER=hipcc
#TBSIZE = 1024 default -- not seeing much difference
# -DDOT_READ_DWORDS_PER_LANE=4

ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//' | tr -d '[:blank:]'`

#hipcc -DTBSIZE=${TBSIZE} -DDOT_READ_DWORDS_PER_LANE=4 \
#   --offload-arch=${ROCM_GPU} -std=c++17 -O3 -DHIP \
#   src/main.cpp src/hip/HIPStream.cpp -o hip-stream -I src/hip -I src

cmake -Bbuild -H. -DMODEL=hip \
      -DCXX_EXTRA_FLAGS="-D__HIP_PLATFORM_AMD__ --offload-arch=${ROCM_GPU} -DTBSIZE=${TBSIZE} -DDOT_READ_DWORDS_PER_LANE=4" \
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

rm -rf ${BABELSTREAM_ROOT}
