import numpy as np
import time

from matrix_prep import prepare_and_transfer, transfer_back_and_free, scale_only_cython

# Pure Python version for comparison
def scale_python(A, scale):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] *= scale

# Benchmark parameters
size = 1000
iterations = 10

# Time pure Python (CPU only)
A = np.random.rand(size, size)
start = time.perf_counter()
for _ in range(iterations):
    scale_python(A.copy(), 2.0)
python_time = (time.perf_counter() - start) / iterations

# Time Cython (CPU only) 
A = np.random.rand(size, size)
start = time.perf_counter()
for _ in range(iterations):
    scale_only_cython(A.copy(), 2.0)
cython_time = (time.perf_counter() - start) / iterations

print(f"\n1. CPU Computation Only ({size}x{size} matrix scaling):")
print(f"   Pure Python: {python_time*1000:.1f} ms")
print(f"   Cython:      {cython_time*1000:.1f} ms")
print(f"   Speedup:     {python_time/cython_time:.1f}x")

# Now show the full pipeline with HIP transfer
print(f"\n2. Full Pipeline (Cython prep + HIP transfer):")
A = np.random.rand(size, size)
start = time.perf_counter()
for _ in range(iterations):
    A_copy = A.copy()
    d_ptr, num_bytes = prepare_and_transfer(A_copy, 2.0)
    transfer_back_and_free(d_ptr, A_copy, num_bytes)
full_time = (time.perf_counter() - start) / iterations
print(f"   Cython + HIP transfer: {full_time*1000:.1f} ms")

# Verify correctness
print(f"\n3. Correctness Check:")
A_py = np.random.rand(10, 10)
A_cy = A_py.copy()
scale_python(A_py, 2.0)
d_ptr, num_bytes = prepare_and_transfer(A_cy, 2.0)
transfer_back_and_free(d_ptr, A_cy, num_bytes)
assert np.allclose(A_py, A_cy), "Results don't match!"
print("   Results match - verified!")

print("\nok")
