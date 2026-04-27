# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrappers around C kernels from HPCTrainingExamples.

Wraps the following repo examples as a shared library:
  - cpu_func     (ManagedMemory/CPU_Code/cpu_code.c)
  - saxpy        (Pragma_Examples/OpenMP/C/1_saxpy)
  - vecadd       (Pragma_Examples/OpenMP/C/3_vecadd)
  - reduction    (Pragma_Examples/OpenMP/C/2_reduction)
"""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DOUBLE_t
ctypedef np.float32_t FLOAT_t


cdef extern from "hpc_kernels.h":
    void cpu_func(double *inp, double *out, int M)
    void saxpy(float a, float *x, float *y, int N)
    void vecadd(double *a, double *b, double *c, int N)
    double reduction(double *x, int n)


# Python wrappers around the c kernels.

def py_cpu_func(np.ndarray[DOUBLE_t, ndim=1] inp):
    """Double every element  (ManagedMemory/CPU_Code/cpu_code.c  cpu_func)."""
    # Ensure correct dtype and contiguous memory to safely take &inp[0]
    inp_c = np.ascontiguousarray(inp, dtype=np.float64)
    cdef int M = inp_c.shape[0]
    if M == 0:
        return np.empty(0, dtype=np.float64)
    cdef np.ndarray[DOUBLE_t, ndim=1] out = np.empty(M, dtype=np.float64)
    cpu_func(&inp_c[0], &out[0], M)
    return out


def py_saxpy(float a,
             np.ndarray[FLOAT_t, ndim=1] x,
             np.ndarray[FLOAT_t, ndim=1] y):
    """y = a*x + y  (Pragma_Examples/OpenMP/C/1_saxpy  saxpy)."""
    # Coerce to float32 contiguous arrays and check lengths
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    y_c = np.ascontiguousarray(y, dtype=np.float32)
    cdef int N = x_c.shape[0]
    if y_c.shape[0] != N:
        raise ValueError("x and y must have the same length")
    if N == 0:
        return y_c.copy()
    cdef np.ndarray[FLOAT_t, ndim=1] y_out = y_c.copy()
    saxpy(a, &x_c[0], &y_out[0], N)
    return y_out


def py_vecadd(np.ndarray[DOUBLE_t, ndim=1] a,
              np.ndarray[DOUBLE_t, ndim=1] b):
    """c = a + b  (Pragma_Examples/OpenMP/C/3_vecadd  vecadd)."""
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    cdef int N = a_c.shape[0]
    if b_c.shape[0] != N:
        raise ValueError("a and b must have the same length")
    if N == 0:
        return np.empty(0, dtype=np.float64)
    cdef np.ndarray[DOUBLE_t, ndim=1] c = np.empty(N, dtype=np.float64)
    vecadd(&a_c[0], &b_c[0], &c[0], N)
    return c


def py_reduction(np.ndarray[DOUBLE_t, ndim=1] x):
    """Sum all elements  (Pragma_Examples/OpenMP/C/2_reduction  reduction)."""
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    cdef int n = x_c.shape[0]
    if n == 0:
        return 0.0
    return reduction(&x_c[0], n)
