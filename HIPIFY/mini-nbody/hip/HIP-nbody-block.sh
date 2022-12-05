#Hipify the blocked cuda source code to hip compatible code
#hipify nbody-block.cu > nbody-block.cpp
#Manually add the first argument onto the kernel argument list
#void bodyForce(Body *p, float dt, int n) //before modification
#void bodyForce(hipLaunchParm lp, Body *p, float dt, int n) //after modification

#compile the hipified source code into executable 
if [ -f nbody-block ]
then
    rm nbody-block
fi

if [ -z  "$HIP_PATH" ]
then

if [ -d /opt/rocm/hip ]
then
    HIP_PATH=/opt/rocm/hip
else
    HIP_PATH=/opt/rocm
fi

fi

echo hipcc -I../ -DSHMOO nbody-block.cpp -o nbody-block
$HIP_PATH/bin/hipcc -I../ -DSHMOO nbody-block.cpp -o nbody-block

#To print our more details, remove DSHMOO flag
#hipcc -I../  nbody-block.cpp -o nbody-block

#execute the program
EXE=nbody-block
K=1024
for i in {1..8}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done

