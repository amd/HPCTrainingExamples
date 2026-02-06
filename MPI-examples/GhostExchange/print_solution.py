# Before running this script
# make sure to run:
# 1. python3 -m venv print_env 
# 2. source print_env/bin/activate
# 3. pip3 install -r requirements.txt 

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

filename = "solution.nc"
vname = "u"
xname = "xcoord"
yname = "ycoord"

png_initial = "initial_solution.png"
png_final   = "final_solution.png"

ds = nc.Dataset(filename, "r")

x = ds[xname][:]  # size: (imax,)
y = ds[yname][:]  # size: (jmax,)
u = ds[vname][:]  # size: (maxIter, jmax, imax)

t0 = 0
tlast = u.shape[0] - 1

X, Y = np.meshgrid(x, y)

def plot_field(field, fname, title):
    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(X, Y, field, shading='auto', cmap='viridis')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    cbar = plt.colorbar(pcm)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

plot_field(u[t0,:,:], png_initial, f"Initial Solution")

plot_field(u[tlast,:,:], png_final, f"Final Solution")

ds.close()
print(f"Saved plots: {png_initial}, {png_final}")

