explore unified shared memory:

explore the behaviour of the examples in this folder with and without HSA_XNACK enabled.
export LIBOMPTARGET_INFO=-1
to see the memomory is moved when you set
export HSA_XNACK=0 
if you set 
export HSA_XNACK=1
the memory copies are gone automatically in example 
cd vector_add_auto_zero_copy 
make
./vector_add_auto_zero_copy
the other example´
cd vector_add_usm
can only run with 
export HSA_XNACK=1
make 
./vector_add_usm
observe the differences in the codes (Hint: vimdiff file1 file2 may help)

The first behaviour is beneficial to have code which is portable between the APU and discrete GPUS. The second example shows the ease of porting (no map clauses or any other data management required) if you start from the CPU and have an APU available to do the porting.
