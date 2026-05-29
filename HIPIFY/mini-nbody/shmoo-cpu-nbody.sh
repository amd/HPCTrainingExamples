SRC=nbody.c
EXE=nbody
gcc -std=gnu99 -O3 -fopenmp -o $EXE $SRC -lm

echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

