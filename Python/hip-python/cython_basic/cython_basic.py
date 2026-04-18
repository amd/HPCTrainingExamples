import numpy as np
import time

# Pure Python version for comparison
def array_sum_python(A):
    result = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result += A[i, j]
    return result

# Benchmark
size = 1000
iterations = 5

A = np.random.rand(size, size)

# Time pure Python
start = time.perf_counter()
for _ in range(iterations):
    result_py = array_sum_python(A)
python_time = (time.perf_counter() - start) / iterations

# Time Cython
from array_sum import array_sum

start = time.perf_counter()
for _ in range(iterations):
    result_cy = array_sum(A)
cython_time = (time.perf_counter() - start) / iterations

print(f"Matrix size: {size}x{size}")
print(f"Pure Python: {python_time*1000:.1f} ms (result: {result_py:.6f})")
print(f"Cython:      {cython_time*1000:.1f} ms (result: {result_cy:.6f})")
print(f"Speedup:     {python_time/cython_time:.1f}x")

# Verify correctness
assert np.isclose(result_py, result_cy), "Results don't match!"
assert np.isclose(result_cy, A.sum()), "Cython result doesn't match numpy!"
print("Correctness verified!")
print("ok")
