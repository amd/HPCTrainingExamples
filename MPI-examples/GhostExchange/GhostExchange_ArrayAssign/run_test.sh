#!/bin/bash
module load cmake rocm/6.0.0 openmpi  omnitrace
#module load aomp-amdclang
module list

# OpenIB is removed as of OpenMPI 5.0.0, so only needed for older versions
CurrentVersion=`mpirun --version |head -1 | tr -d '[:alpha:] ) (' `
RequiredVersion="5.0.0"
if [ "$(printf '%s\n' "$RequiredVersion" "$CurrentVersion" | sort -Vr | head -n1)" = "$RequiredVersion" ]; then 
   echo "Setting MPIRUN options to exclude openib transport layer for mpi version ${CurrentVersion}"
   echo "OpenMPI versions starting with 5.0.0 have the legacy openib transport layer removed"
   MPI_RUN_OPTIONS="--mca pml ob1 --mca btl ^openib"
fi

# Setting HSA_XNACK now for all of the following runs
export HSA_XNACK=1

cd Orig
rm -rf build
mkdir build && cd build
cmake ..
make
mpirun ${MPI_RUN_OPTIONS} -n 16  ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
#mpirun ${MPI_RUN_OPTIONS} -n 64  ./GhostExchange -x 8  -y 8  -i 20000 -j 20000 -h 2 -t -c
#mpirun ${MPI_RUN_OPTIONS} -n 256 ./GhostExchange -x 16 -y 16 -i 20000 -j 20000 -h 2 -t -c
#mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
#mpirun ${MPI_RUN_OPTIONS} -n 64  --bind-to core     -map-by ppr:8:numa  ./GhostExchange -x 8  -y 8  -i 20000 -j 20000 -h 2 -t -c
#mpirun ${MPI_RUN_OPTIONS} -n 256 --bind-to hwthread -map-by ppr:32:numa ./GhostExchange -x 16 -y 16 -i 20000 -j 20000 -h 2 -t -c
cd ../..

echo "Running Ver1"
cd Ver1
rm -rf build
mkdir build && cd build
cmake ..
make
mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
#mpirun -n 64  --bind-to core     -map-by ppr:8:numa  ./GhostExchange -x 8  -y 8  -i 20000 -j 20000 -h 2 -t -c
#mpirun -n 256 --bind-to hwthread -map-by ppr:32:numa ./GhostExchange -x 16 -y 16 -i 20000 -j 20000 -h 2 -t -c
cd ../..

cd Ver2
rm -rf build
mkdir build && cd build
cmake ..
make
export LIBOMPTARGET_INFO=-1
#export LIBOMPTARGET_KERNEL_TRACE=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
export OMNITRACE_USE_PROCESS_SAMPLING=false
export OMP_NUM_THREADS=1
omnitrace-instrument -o GhostExchange.inst -- GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  ../../affinity_script.sh omnitrace-run -- ./GhostExchange.inst -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
unset LIBOMPTARGET_INFO
unset LIBOMPTARGET_KERNEL_TRACE
cd ../..

echo "Running Ver3"
cd Ver3
rm -rf build
mkdir build && cd build
cmake ..
make
mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c 
cd ../..

echo "Running Ver4"
cd Ver4
rm -rf build
mkdir build && cd build
cmake ..
make
mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c 
cd ../..

export HSA_XNACK=0

echo "Running Ver5"
cd Ver5
rm -rf build
mkdir build && cd build
cmake ..
make
mpirun ${MPI_RUN_OPTIONS} -n 16  --bind-to core     -map-by ppr:2:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c 

cd ../..
