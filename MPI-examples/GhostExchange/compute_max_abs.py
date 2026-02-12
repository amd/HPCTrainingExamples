import netCDF4 as nc
import numpy as np

ds = nc.Dataset('diff.nc', 'r')
u = ds.variables['u'][:]
ds.close()

u_flat = u.flatten()

u_valid = u_flat[np.abs(u_flat) < 1e36]

if len(u_valid) > 0:
    max_abs = np.max(np.abs(u_valid))
    print(f"Maximum absolute value: {max_abs}")
else:
    print("No valid values found!")
    max_abs = 0.0

out = nc.Dataset('out.nc', 'w')
out.createDimension('scalar', 1)
var = out.createVariable('u_max', 'f8', ('scalar',))
var[0] = max_abs
out.close()

print(f"Result written to out.nc: {max_abs}")

