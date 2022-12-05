SRC=../nbody-orig.c
EXE=nbody-orig-mic
MICROOT=/shared/apps/rhel-6.2/intel/ics-2013/composerxe/lib/mic
MIC=mic0
if [ $# -gt 0 ]
  then
    MIC=$1
fi

icc -std=c99 -openmp -mmic -DSHMOO -o $EXE $SRC

scp $EXE $MIC:~/
scp $MICROOT/libiomp5.so $MIC:~/

echo $EXE

K=1024
for i in {1..10}
do
    ssh $MIC "export LD_LIBRARY_PATH=~/:$LD_LIBRARY_PATH; ./$EXE $K"
    K=$(($K*2))
done

