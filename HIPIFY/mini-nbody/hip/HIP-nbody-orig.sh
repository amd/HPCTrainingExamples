#Hipify the original cuda source code to hip compatible code
#hipify ../cuda/nbody-orig.cu > nbody-orig.hip

#compile the hipified source code into executable 
if [ -f nbody-orig ]
then
    rm nbody-orig
fi

echo hipcc -I../ -DSHMOO nbody-orig.hip -o nbody-orig
$ROCM_PATH/bin/hipcc -I../ -DSHMOO nbody-orig.hip -o nbody-orig

#To print our more details, remove  flag
#hipcc -I../  nbody-orig.hip -o nbody-orig

#execute the program

EXE=nbody-orig
K=1024
for i in {1..10}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done

