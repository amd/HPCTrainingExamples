import cupy as cp
import numpy as np
import xarray as xr
import cupy_xarray  # adds .cupy to Xarray objects

# create an array on the CPU with NumPy
arr_cpu = np.random.rand(10, 10, 10)

# create an array on the GPU with CuPy
arr_gpu = cp.random.rand(10, 10, 10)

# create a DataArray using NumPy array with three dimensions and 10 elements along each dimension
da_np = xr.DataArray(arr_cpu, dims=["x", "y", "time"])

# create a DataArray using CuPy array with three dimensions and 10 elements along each dimension
da_cp = xr.DataArray(arr_gpu, dims=["x", "y", "time"])

print("Is the array used to create da_np on device?", da_np.cupy.is_cupy)
print("Is the array used to create da_cp on device?", da_cp.cupy.is_cupy)

# access the underlying CuPy array used to create the xarray.DataArray
cupy_array = da_cp.data

print("da_cp.data is of type:", type(cupy_array))

# check that the array used to create the xarray and the one given by xarray are the same
print("check that arr_gpu and cupy_array are the same with CuPy:", cp.allclose(cupy_array,arr_gpu))

# check that the array used to create the xarray and the one given by xarray are the same
print("check the arr_gpu and cupy_array are the same with NumPy (interoperability):", np.allclose(cupy_array,arr_gpu))

# print device on which array exists
print("arr_gpu is on device:", arr_gpu.device)
print("arr_cpu is on device:", arr_cpu.device)

# print total number of available devices
print("total number of available devices:", cp.cuda.runtime.getDeviceCount())

# use the device context manager to create data on a different device
with cp.cuda.Device(2):
    arr_gpu2 = cp.array([1, 2, 3, 4, 5])
print("arr_gpu2 is on device:", arr_gpu2.device)

