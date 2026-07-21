#Hipify the soa cuda source code to hip compatible code
#hipify nbody-soa.cu > nbody-soa.hip
#Manually add the first argument onto the kernel argument list
#void bodyForce(Body *p, float dt, int n) //before modification
#void bodyForce(hipLaunchParm lp, Body *p, float dt, int n) //after modification

#compile the hipified source code into executable 
if [ -f nbody-soa ]
then
    rm nbody-soa
fi

echo hipcc -I../ nbody-soa.hip -o nbody-soa
$ROCM_PATH/bin/hipcc -I../ nbody-soa.hip -o nbody-soa

#execute the program
EXE=nbody-soa
K=1024
for i in {1..8}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done

