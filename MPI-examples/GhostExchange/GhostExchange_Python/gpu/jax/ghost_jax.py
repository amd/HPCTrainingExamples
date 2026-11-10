import os
import sys
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
try:
    shard_map = jax.shard_map
except AttributeError:  # older jax
    from jax.experimental.shard_map import shard_map

def parse_input_args(argv, jmax, imax, nprocy, nprocx, maxIter, do_print):
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '-c':
            i += 1
        elif a == '-t':
            i += 1
        elif a == '-p':
            do_print = 1
            i += 1
        elif a == '-h':
            i += 2
        elif a == '-I':
            maxIter = int(argv[i + 1]); i += 2
        elif a == '-i':
            imax = int(argv[i + 1]); i += 2
        elif a == '-j':
            jmax = int(argv[i + 1]); i += 2
        elif a == '-x':
            nprocx = int(argv[i + 1]); i += 2
        elif a == '-y':
            nprocy = int(argv[i + 1]); i += 2
        else:
            print(f"Unknown option {a}", file=sys.stderr); sys.exit(1)
    return jmax, imax, nprocy, nprocx, maxIter, do_print


def make_step(mesh, nprocx, nprocy):
    
    def jacobi_local(u):
        xi = jax.lax.axis_index('x')
        yi = jax.lax.axis_index('y')

        # --- x halos (columns) ---
        if nprocx > 1:
            lh = jax.lax.ppermute(u[:, -1:], 'x', [(k, k + 1) for k in range(nprocx - 1)])
            rh = jax.lax.ppermute(u[:, :1], 'x', [(k, k - 1) for k in range(1, nprocx)])
        else:
            lh = u[:, -1:]
            rh = u[:, :1]
        left_halo = jnp.where(xi == 0, u[:, :1], lh)
        right_halo = jnp.where(xi == nprocx - 1, u[:, -1:], rh)
        west = jnp.concatenate([left_halo, u[:, :-1]], axis=1)   # u[:, i-1]
        east = jnp.concatenate([u[:, 1:], right_halo], axis=1)   # u[:, i+1]

        # --- y halos (rows) ---
        if nprocy > 1:
            uh = jax.lax.ppermute(u[-1:, :], 'y', [(k, k + 1) for k in range(nprocy - 1)])
            dh = jax.lax.ppermute(u[:1, :], 'y', [(k, k - 1) for k in range(1, nprocy)])
        else:
            uh = u[-1:, :]
            dh = u[:1, :]
        up_halo = jnp.where(yi == 0, u[:1, :], uh)
        down_halo = jnp.where(yi == nprocy - 1, u[-1:, :], dh)
        north = jnp.concatenate([up_halo, u[:-1, :]], axis=0)    # u[j-1, :]
        south = jnp.concatenate([u[1:, :], down_halo], axis=0)   # u[j+1, :]

        return (u + west + east + north + south) / 5.0

    step = shard_map(jacobi_local, mesh=mesh,
                     in_specs=P('y', 'x'), out_specs=P('y', 'x'))

    @jax.jit
    def run(u, n):
        return jax.lax.fori_loop(0, n, lambda i, s: step(s), u)

    return run


def write_netcdf_soln(fname, u_global, imax, jmax):
    from netCDF4 import Dataset
    ncid = Dataset(fname, 'w', format='NETCDF4')
    ncid.createDimension('time', None)
    ncid.createDimension('y', jmax)
    ncid.createDimension('x', imax)
    varid = ncid.createVariable('u', 'f8', ('time', 'y', 'x'))
    vx = ncid.createVariable('xcoord', 'f8', ('x',))
    vy = ncid.createVariable('ycoord', 'f8', ('y',))
    varid.description = "solution field"
    # flip y to match the other variants' output orientation
    varid[0, :, :] = np.asarray(u_global)[::-1, :]
    vx[:] = np.linspace(0.0, (imax - 1) / imax, imax)
    vy[:] = np.linspace(0.0, (jmax - 1) / jmax, jmax)
    ncid.close()


def main():
    # One process per GPU (srun) -> distributed; single process -> local GPUs.
    nproc_env = int(os.environ.get('SLURM_NTASKS', '1'))
    if nproc_env > 1:
        jax.distributed.initialize()

    rank = jax.process_index()

    imax = 2000
    jmax = 2000
    nprocx = 0
    nprocy = 0
    maxIter = 1000
    do_print = 0
    jmax, imax, nprocy, nprocx, maxIter, do_print = \
        parse_input_args(sys.argv, jmax, imax, nprocy, nprocx, maxIter, do_print)

    if rank == 0:
        print("------> Initializing the Problem")

    tstart_total = time.time()

    devices = jax.devices()
    if nprocx * nprocy != len(devices):
        if rank == 0:
            print(f"error: -x {nprocx} * -y {nprocy} = {nprocx*nprocy} != "
                  f"{len(devices)} devices", file=sys.stderr)
        sys.exit(1)
    mesh = Mesh(np.asarray(devices).reshape(nprocy, nprocx), axis_names=('y', 'x'))
    sharding = NamedSharding(mesh, P('y', 'x'))

    # Gaussian initial condition (index space), matching the other variants.
    Lx = Ly = 1.0
    sigma = 0.01 / ((Lx / imax) * (Ly / jmax))
    x_center = imax / 2.0
    y_center = jmax / 2.0

    def init_cb(index):
        jsl, isl = index
        j0, j1, _ = jsl.indices(jmax)   # resolves full-axis slices (start=None)
        i0, i1, _ = isl.indices(imax)
        jj = np.arange(j0, j1)
        ii = np.arange(i0, i1)
        J, I = np.meshgrid(jj, ii, indexing='ij')
        return np.exp(-0.5 * (((I - x_center) ** 2) / sigma ** 2 +
                              ((J - y_center) ** 2) / sigma ** 2)).astype(np.float64)

    u = jax.make_array_from_callback((jmax, imax), sharding, init_cb)

    run = make_step(mesh, nprocx, nprocy)

    # compile / warm up (not timed). Use a throwaway so the real solution is
    # NOT advanced here: the timed run below must do exactly maxIter steps.
    warm = run(u, 1)
    warm.block_until_ready()
    del warm

    if rank == 0:
        print("------> Advancing the Solution\n")

    tstart_solve = time.time()
    u = run(u, maxIter)
    u.block_until_ready()
    solve_time = time.time() - tstart_solve

    # global checksum (decomposition-invariant), for correctness checks
    gsum = float(jnp.sum(u))
    gmax = float(jnp.max(u))

    if do_print:
        from jax.experimental import multihost_utils
        u_global = multihost_utils.process_allgather(u, tiled=True) if nproc_env > 1 else np.asarray(u)
        if rank == 0:
            write_netcdf_soln("solution.nc", u_global, imax, jmax)

    total_time = time.time() - tstart_total

    if rank == 0:
        print("------> Printing Timings")
        print(f"        Solution Advancement: {solve_time:.6f}")
        print(f"        Total: {total_time:.6f}")
        print(f"        Checksum: sum={gsum:.10e} max={gmax:.10e}")
        print("        (ghost/halo exchange is fused into the stencil by XLA as a "
              "collective-permute; on ROCm this lowers to RCCL)")


if __name__ == "__main__":
    main()
