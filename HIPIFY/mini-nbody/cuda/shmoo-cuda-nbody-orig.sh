SRC=nbody-orig.cu
EXE=nbody-orig

nvcc -arch=sm_35 -I../ -o $EXE $SRC

echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

