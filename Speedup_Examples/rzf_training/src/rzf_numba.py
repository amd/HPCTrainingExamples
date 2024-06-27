#!/usr/bin/env -S python3

import math
from numba import njit

@njit
def zeta():
    s=2
    Nmax = 10000000000
    _zeta = 0.0

    for n in range(1,Nmax):
        _zeta += (1.0*n)**(-s)
    return _zeta

z = zeta()
print(f"z={z}\n")
pi = math.sqrt(6.0 * z)
print(f"zeta = {z}\npi = {pi}\n")
