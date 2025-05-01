## MPI4Py examples

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.


### Exploring MPI communication with MPI4Py

First set up the environment

```
module load mpi4py cupy
```

Add `print("Rank is:", rank)` right after the rank is set at line 10.

Then run the python program

```
mpirun -n 4 python mpi4py_cupy.py
```

You should see the following output, but it might be in a different order

```
Rank is: 3
Rank is: 1
Rank is: 2
Rank is: 0
Starting allreduce test...
Starting bcast test...
Starting send-recv test...
Success
```

To verify if the program is running on the GPU

```
export AMD_LOG_LEVEL=3
mpirun -n 4 python mpi4py_cupy.py
```

You will get a lot of output including whole programs.
