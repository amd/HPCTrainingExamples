# Test contributed by Corey Adams (corey.adams@amd.com)

# This test checks that JAX
# shows the expected performance
# for different data types

import time

import jax
from jax import numpy
from jax import random
from jax import jit

# JAX requires special considerations to enable fp64:
from jax import config
config.update("jax_enable_x64", True)

# Define two matrices for multiplication:
# (make them pretty big!)

n = 8192
m = 8192
k = 16384

required_ops = n * m * (2*k-1) / 1000**4 # TFlops


key = random.PRNGKey(0)


# Start with matrices in fp64 for a high precision matmul
key, subkey = random.split(key)
A = random.normal(subkey, (n, k),  dtype=numpy.float64)/k

key, subkey = random.split(key)
B = random.normal(subkey, (k, m),  dtype=numpy.float64)/k


C = numpy.matmul(A, B)


# For JAX, we want to enable XLA compilation of the matmul so put this into a function:
@jit
def jax_matmul(A, B):
    return numpy.matmul(A, B)

# We will do the benchmarks ourself:
def benchmark(A, B, n = 10):
    times = []
    for i in range(n):
        start = time.time()
        C = jax_matmul(A, B)
        # For JAX to wait for the result:
        C.block_until_ready()
        times.append(time.time() - start)
    return numpy.median(numpy.asarray(times))

m_fp64 = benchmark(A, B)

print(f"Acheived {A.dtype} TFLOPS: {required_ops / m_fp64 :.3f}")

times = {}

# 32 and 16 bit ops:
for dtype in (numpy.float32, numpy.bfloat16, numpy.float16, numpy.float8_e4m3fnuz):

    A_casted = A.astype(dtype)
    B_casted = B.astype(dtype)


    m_reduced = benchmark(A_casted, B_casted)

    print(f"Acheived {A_casted.dtype} TFLOPS: {required_ops / m_reduced :.3f}")
