SRC=nbody-soa.cu
EXE=nbody-soa

nvcc -arch=sm_35 -I../ -DSHMOO -o $EXE $SRC

echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

