# Ghost exchange example in Python

Make sure to load the necessary modules:

```
module load rocm openmpi hdf5 netcdf-c mpi4py
```

Note that `hdf5` has to be built with parallel support based on the `openmpi` installation and `netcdf-c` has to be built using this parallel enabled `hdf5`. Furthermore, `mpi4py` also needs to be built using the same `openmpi`.

Scripts to build this software from source can be found in our [HPCTrainingDock](https://github.com/amd/HPCTrainingDock) repo.

Then, create and activate a virtual environment, with a couple of dependencies for the installation of Python module of netCDF4:

```
python3 -m venv ghost_exchange_env
source ghost_exchange_env/bin/activate
pip3 install cython setuptools
```

Next, clone the `netcdf4-python` repo and install from source:

```
git clone https://github.com/Unidata/netcdf4-python.git
cd netcdf4-python
export HDF5_DIR=$HDF5_ROOT
export NETCDF4_DIR=$NETCDF_C_ROOT
python3 setup.py build
pip3 install .
```

Then run with:
```
mpirun -n 4 python3 orig.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

Note that unlike the C++ code, the `-p` option here is used to create the netCDF output file called `solution.nc`. Without the `-p` option no I/O is performed.

