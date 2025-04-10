import cupy as cp
import numpy as np

# Create a CuPy array
gpu_array = cp.array([1, 2, 3, 4, 5])
print("CuPy Array:", gpu_array)


# Perform operations on the GPU​
gpu_array_squared = gpu_array ** 2
print("Squared CuPy Array:", gpu_array_squared)


# Create a NumPy array​
cpu_array = np.array([5, 6, 7, 8, 9])
print("NumPy Array:", cpu_array)


# Transfer NumPy array to GPU​
gpu_array_from_cpu = cp.asarray(cpu_array)
print("CuPy Array from NumPy:", gpu_array_from_cpu)


# Perform element-wise addition​
result_gpu = gpu_array + gpu_array_from_cpu
print("Addition Result on GPU:", result_gpu)


# Transfer result back to CPU
result_cpu = cp.asnumpy(result_gpu)
print("Result on CPU:", result_cpu)
