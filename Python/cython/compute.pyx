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

# ── C declarations (from hpc_kernels.h) ──────────────────────────
cdef extern from "hpc_kernels.h":
    void cpu_func(double *inp, double *out, int M)
    void saxpy(float a, float *x, float *y, int N)
    void vecadd(double *a, double *b, double *c, int N)
    double reduction(double *x, int n)


# ── Python-visible wrappers ──────────────────────────────────────

def py_cpu_func(np.ndarray[DOUBLE_t, ndim=1] inp):
    """Double every element  (ManagedMemory/CPU_Code/cpu_code.c  cpu_func)."""
    cdef int M = inp.shape[0]
    cdef np.ndarray[DOUBLE_t, ndim=1] out = np.empty(M, dtype=np.float64)
    cpu_func(&inp[0], &out[0], M)
    return out


def py_saxpy(float a,
             np.ndarray[FLOAT_t, ndim=1] x,
             np.ndarray[FLOAT_t, ndim=1] y):
    """y = a*x + y  (Pragma_Examples/OpenMP/C/1_saxpy  saxpy)."""
    cdef int N = x.shape[0]
    cdef np.ndarray[FLOAT_t, ndim=1] y_out = y.copy()
    saxpy(a, &x[0], &y_out[0], N)
    return y_out


def py_vecadd(np.ndarray[DOUBLE_t, ndim=1] a,
              np.ndarray[DOUBLE_t, ndim=1] b):
    """c = a + b  (Pragma_Examples/OpenMP/C/3_vecadd  vecadd)."""
    cdef int N = a.shape[0]
    cdef np.ndarray[DOUBLE_t, ndim=1] c = np.empty(N, dtype=np.float64)
    vecadd(&a[0], &b[0], &c[0], N)
    return c


def py_reduction(np.ndarray[DOUBLE_t, ndim=1] x):
    """Sum all elements  (Pragma_Examples/OpenMP/C/2_reduction  reduction)."""
    cdef int n = x.shape[0]
    return reduction(&x[0], n)
