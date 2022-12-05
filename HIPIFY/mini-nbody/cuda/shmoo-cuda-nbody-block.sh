SRC=nbody-block.cu
EXE=nbody-block

nvcc -arch=sm_35 -ftz=true -I../ -o $EXE $SRC -DSHMOO

echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

