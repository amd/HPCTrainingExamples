def array_sum(double[:, ::1] A):
    """Compute the sum of all elements in a 2D array."""
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int i, j
    cdef double result = 0

    for i in range(m):
        for j in range(n):
            result += A[i, j]

    return result
