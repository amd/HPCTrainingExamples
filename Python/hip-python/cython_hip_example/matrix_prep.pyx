# cython: language_level=3
cimport cython
import numpy as np
from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def scale_only_cython(double[:, ::1] A, double scale):
    """
    Cython-optimized matrix scaling (CPU only).
    Used to benchmark Cython speedup separately from HIP transfer.
    """
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int i, j
    
    for i in range(m):
        for j in range(n):
            A[i, j] *= scale

@cython.boundscheck(False)
@cython.wraparound(False)
def prepare_and_transfer(double[:, ::1] A, double scale):
    """
    Cython-optimized: scale matrix on CPU, then transfer to GPU.
    The nested loop is what Cython accelerates vs pure Python.
    """
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int i, j
    
    # CPU work: scale the matrix (Cython makes this fast)
    for i in range(m):
        for j in range(n):
            A[i, j] *= scale
    
    # GPU work: transfer to device
    num_bytes = A.shape[0] * A.shape[1] * sizeof(double)
    d_ptr = hip_check(hip.hipMalloc(num_bytes))
    hip_check(hip.hipMemcpy(d_ptr, A, num_bytes, 
                            hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    return d_ptr, num_bytes

def transfer_back_and_free(d_ptr, double[:, ::1] A, size_t num_bytes):
    """Copy results from GPU and free device memory."""
    hip_check(hip.hipMemcpy(A, d_ptr, num_bytes,
                            hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipFree(d_ptr))
