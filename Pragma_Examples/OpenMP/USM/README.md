# Explore Unified Shared Memory with OpenMP

Explore the behavior of the examples in this folder with and without `HSA_XNACK` enabled. Use:

```
export LIBOMPTARGET_INFO=-1
```
to see if the memory is moved when you set

```
export HSA_XNACK=0
```

If you set
```
export HSA_XNACK=1
```
the memory copies are gone automatically in the `vector_add_auto_zero_copy` example:

To compile and run:

```
cd vector_add_auto_zero_copy
make
./vector_add_auto_zero_copy
```


To compile and run the other example:

```
cd vector_add_usm
// can only run with
export HSA_XNACK=1
make
./vector_add_usm
```
Note, this can only run with `HSA_XNACK=1`. Observe the differences in the codes (Hint: `vimdiff file1 file2` may help).

The first example shows code that is portable between the APU and discrete GPUS. The second example shows the ease of porting (no map clauses or any other data management required) if you start from the CPU and have an APU available to do the porting.
