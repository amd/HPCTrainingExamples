#!/usr/bin/env python3
"""Compare the solution.nc files written by two GhostExchange runs (-p option).

    python3 compare_solutions.py solution_a.nc solution_b.nc

Compares the first record (initial condition) and the last record (final field)
of the variable `u` and prints the maximum absolute difference. A value at the
level of floating-point roundoff (about 1e-15 here) means the solutions match.
Exit status: 0 on a match, 1 otherwise. Needs only netCDF4 and numpy (no NCO).
Both runs must use the same -i, -j and -I.
"""
import sys
import numpy as np
import netCDF4 as nc

TOL = 1e-12   # roundoff-level agreement for fp64


def load(path):
    # Read only the first and last record: the files have maxIter+1 records,
    # almost all of them never written (fill values).
    try:
        ds = nc.Dataset(path)
    except OSError as e:
        sys.exit(f"cannot open {path} as netCDF ({e}); "
                 "pass the solution.nc written by a run with -p")
    with ds:
        u = ds.variables['u']
        nt = u.shape[0]
        return nt, u.shape[1:], u[0], u[nt - 1]


if len(sys.argv) != 3:
    sys.exit(__doc__)

pa, pb = sys.argv[1:3]
nta, sha, a0, a1 = load(pa)
ntb, shb, b0, b1 = load(pb)
print(f"A = {pa}: {nta} records of {sha[0]}x{sha[1]}")
print(f"B = {pb}: {ntb} records of {shb[0]}x{shb[1]}")
if sha != shb:
    sys.exit("grid sizes differ: same -i/-j in both runs?")
if nta != ntb:
    sys.exit("record counts differ: same -I in both runs?")

ok = True


def compare(label, x, y):
    global ok
    x = np.ma.filled(x, np.nan)
    y = np.ma.filled(y, np.nan)
    d = np.abs(x - y)
    bad = np.count_nonzero(~np.isfinite(d))
    if bad:
        print(f"{label}: {bad} non-finite values -> FAIL")
        ok = False
        return
    j, i = np.unravel_index(np.argmax(d), d.shape)
    m = d[j, i]
    verdict = 'match' if m < TOL else 'DIFFER'
    print(f"{label}: max|A-B| = {m:.3e} at (y={j}, x={i})  -> {verdict}")
    ok = ok and bool(m < TOL)


compare("initial (record 0)", a0, b0)
compare(f"final   (record {nta - 1})", a1, b1)
sys.exit(0 if ok else 1)
