#!/bin/sh
for dir in OpenMP/C/Make/vecadd OpenMP/Fortran/Make/vecadd OpenMP/Fortran/Make/freduce OpenACC/C/Make/vecadd OpenACC/Fortran/Make/vecadd OpenACC/Fortran/Make/freduce OpenMP/C/Make/saxpy OpenACC/C/Make/saxpy OpenMP/C/Make/reduction OpenACC/C/Make/reduction
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
