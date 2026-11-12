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


def world_size():
    # Number of launched processes, across common HPC launchers
    # (Slurm srun, Open MPI / MPICH mpirun, torchrun-style WORLD_SIZE).
    for var in ('SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE',
                'WORLD_SIZE'):
        val = os.environ.get(var)
        if val:
            return int(val)
    return 1


def init_distributed():
    # Bring up the JAX distributed runtime for a multi-process run.
    #
    # JAX uses one global device mesh and the *device* (not the process) is the
    # unit of parallelism, so a single process can drive several local GPUs.
    # Two launch layouts are supported and give the identical computation
    # (same global device set, same mesh):
    #   * one process per node  -> each process drives all its local GPUs
    #     (recommended on Slurm: robust, no per-task GPU binding needed)
    #         srun -N<nodes> --ntasks-per-node=1 python ghost_jax.py ...
    #   * one process per GPU   -> each process drives a single GPU
    #         srun -N<nodes> --ntasks-per-node=<gpus-per-node> python ghost_jax.py ...
    #
    # Some clusters export HTTP(S) proxies that also intercept the coordinator's
    # gRPC traffic and make initialize() hang/time out; the coordinator is always
    # reached directly inside the cluster, so clear proxy settings first.
    for _p in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
               'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
        os.environ.pop(_p, None)

    # Non-Slurm launchers (Open MPI, cloud, ...): let JAX auto-detect the cluster.
    if 'SLURM_PROCID' not in os.environ:
        jax.distributed.initialize()
        return

    # Slurm: set the topology explicitly (auto-detection is brittle when a
    # process owns more than one GPU, or when GPU binding is not per-task).
    import subprocess
    ntasks = int(os.environ['SLURM_NTASKS'])
    procid = int(os.environ['SLURM_PROCID'])
    localid = int(os.environ.get('SLURM_LOCALID', '0'))
    nnodes = int(os.environ.get('SLURM_NNODES', '1'))

    coord = os.environ.get('JAX_COORDINATOR_ADDRESS')
    if not coord:
        nodelist = (os.environ.get('SLURM_JOB_NODELIST')
                    or os.environ.get('SLURM_NODELIST', ''))
        first = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', nodelist]).decode().split()[0]
        coord = f"{first}:{os.environ.get('JAX_COORDINATOR_PORT', '12421')}"

    kwargs = dict(coordinator_address=coord, num_processes=ntasks,
                  process_id=procid, initialization_timeout=600)

    # Assign this process the block of local GPUs it should drive. Works for
    # both layouts: tasks-per-node=1 -> all local GPUs; tasks-per-node=G -> 1 each.
    tpn_env = os.environ.get('SLURM_NTASKS_PER_NODE', '').split('(')[0]
    tasks_per_node = int(tpn_env) if tpn_env else max(1, ntasks // nnodes)
    gpus_on_node = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
    if gpus_on_node:
        per_task = max(1, gpus_on_node // tasks_per_node)
        kwargs['local_device_ids'] = list(range(localid * per_task,
                                                 (localid + 1) * per_task))
    jax.distributed.initialize(**kwargs)


def main():
    nproc = world_size()
    if nproc > 1:
        init_distributed()

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
        u_global = multihost_utils.process_allgather(u, tiled=True) if nproc > 1 else np.asarray(u)
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
