import numpy as np
import time
from numba import hip
from hip import hip as hiprt

# JIT-compile kernel for AMD GPUs via native numba.hip API
@hip.jit
def vec_add(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = hip.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]


# Create host arrays
N = 1024
a_host = np.arange(N, dtype=np.float32)
b_host = np.arange(N, dtype=np.float32) * 2
c_host = np.zeros(N, dtype=np.float32)

# Transfer to GPU memory via Numba-HIP API
a_dev = hip.to_device(a_host)
b_dev = hip.to_device(b_host)
c_dev = hip.to_device(c_host)

# Launch config: ceil(N / threads_per_block) blocks
threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block

# Warm-up run (JIT compilation happens here)
vec_add[blocks, threads_per_block](a_dev, b_dev, c_dev)
hiprt.hipDeviceSynchronize()

# Timed runs
n_runs = 5
start = time.perf_counter()
for i in range(n_runs):
    vec_add[blocks, threads_per_block](a_dev, b_dev, c_dev)
    hiprt.hipDeviceSynchronize()
elapsed = time.perf_counter() - start
print(f"GPU: {n_runs} runs in {elapsed*1e6:.1f} us ({elapsed/n_runs*1e6:.1f} us/run)")

# Copy result back and verify
c_host = c_dev.copy_to_host()
expected = a_host + b_host
assert np.allclose(c_host, expected), "Result mismatch!"
print(f"PASSED")
