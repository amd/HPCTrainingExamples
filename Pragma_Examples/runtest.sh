#!/bin/bash
module load aomp
for dir in OpenMP/C/vecadd OpenMP/Fortran/vecadd OpenMP/Fortran/freduce OpenACC/C/vecadd OpenACC/Fortran/vecadd OpenACC/Fortran/freduce OpenMP/C/saxpy OpenACC/C/saxpy OpenMP/C/reduction OpenACC/C/reduction
do
  pushd $dir
  make
  executable=$(basename $dir)
  if [ -x ./$executable ] 
  then
     echo "Compile succeded" > result.txt
     ./$executable 2> ${executable}_stderr.out | tee ${executable}_run.out
     if [ `grep "HSA run-time initialized for GCN" ${executable}_stderr.out | wc -l` == 1 ]; then
        echo "Ran on the GPU" >> result.txt
     elif [ `grep "Entering OpenMP kernel" ${executable}_stderr.out | wc -l` -ge 1 ]; then
        echo "Ran on the GPU" >> result.txt
     elif [ `grep DEVID ${executable}_stderr.out | wc -l` == 1 ]; then
        echo "Ran on the GPU" >> result.txt
     else
        echo "Did not run on the GPU" >> result.txt
     fi
  else
     echo "Compile failed" > result.txt
  fi
  make clean
  popd
done
