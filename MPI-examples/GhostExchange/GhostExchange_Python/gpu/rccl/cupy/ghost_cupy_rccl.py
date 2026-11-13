from mpi4py import MPI
from netCDF4 import Dataset
import numpy as np
import cupy as cp
from cupy.cuda import nccl
import time
import sys

boundarycondition_time = 0.0
ghostcell_time = 0.0

NCCL_F64 = nccl.NCCL_FLOAT64


def parse_input_args(argv, jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    i = 1
    while i < len(argv):
        if argv[i] == '-c':
            corners = 1
            i += 1
        elif argv[i] == '-t':
            do_timing = 1
            i += 1
        elif argv[i] == '-p':
            do_print = 1
            i += 1
        elif argv[i] == '-h':
            if i + 1 < len(argv):
                nhalo = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -h requires an argument.", file=sys.stderr)
                sys.exit(1)
        elif argv[i] == '-I':
            if i + 1 < len(argv):
                maxIter = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -I requires an argument.", file=sys.stderr)
                sys.exit(1)
        elif argv[i] == '-i':
            if i + 1 < len(argv):
                imax = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -i requires an argument.", file=sys.stderr)
                sys.exit(1)
        elif argv[i] == '-j':
            if i + 1 < len(argv):
                jmax = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -j requires an argument.", file=sys.stderr)
                sys.exit(1)
        elif argv[i] == '-x':
            if i + 1 < len(argv):
                nprocx = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -x requires an argument.", file=sys.stderr)
                sys.exit(1)
        elif argv[i] == '-y':
            if i + 1 < len(argv):
                nprocy = int(argv[i + 1])
                i += 2
            else:
                if rank == 0:
                    print(f"Option -y requires an argument.", file=sys.stderr)
                sys.exit(1)
        else:
            if rank == 0:
                print(f"Unknown option {argv[i]}", file=sys.stderr)
            sys.exit(1)

    return jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print


def boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop):

    global boundarycondition_time

    tstart_boundarycondition = time.time()

    if nleft == MPI.PROC_NULL:
        x[nhalo:jsize+nhalo, 0:nhalo] = x[nhalo:jsize+nhalo, nhalo:nhalo+1]

    if nrght == MPI.PROC_NULL:
        x[nhalo:jsize+nhalo, isize+nhalo:isize+nhalo+nhalo] = x[nhalo:jsize+nhalo, isize-1+nhalo:isize+nhalo]

    if nbot == MPI.PROC_NULL:
        x[0:nhalo, 0:isize+nhalo+nhalo] = x[nhalo:nhalo+1, 0:isize+nhalo+nhalo]

    if ntop == MPI.PROC_NULL:
        x[jsize+nhalo:jsize+nhalo+nhalo, 0:isize+nhalo+nhalo] = x[jsize-1+nhalo:jsize+nhalo, 0:isize+nhalo+nhalo]

    boundarycondition_time += time.time() - tstart_boundarycondition


def ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, nc, stream, do_timing):

    global ghostcell_time

    tstart_ghostcell = time.time()

    jlow = nhalo
    jhgh = nhalo + jsize
    if corners:
        if nbot == MPI.PROC_NULL:
            jlow = 0
        if ntop == MPI.PROC_NULL:
            jhgh = nhalo + jsize + nhalo

    jnum = jhgh - jlow

    # --- left/right exchange (halo columns) ---
    left_send = cp.ascontiguousarray(x[jlow:jhgh, nhalo:2*nhalo]) if nleft != MPI.PROC_NULL else None
    rght_send = cp.ascontiguousarray(x[jlow:jhgh, isize:nhalo+isize]) if nrght != MPI.PROC_NULL else None
    left_recv = cp.empty((jnum, nhalo), dtype=np.float64) if nleft != MPI.PROC_NULL else None
    rght_recv = cp.empty((jnum, nhalo), dtype=np.float64) if nrght != MPI.PROC_NULL else None

    nccl.groupStart()
    if nrght != MPI.PROC_NULL:
        nc.recv(rght_recv.data.ptr, rght_recv.size, NCCL_F64, nrght, stream.ptr)
    if nleft != MPI.PROC_NULL:
        nc.recv(left_recv.data.ptr, left_recv.size, NCCL_F64, nleft, stream.ptr)
    if nleft != MPI.PROC_NULL:
        nc.send(left_send.data.ptr, left_send.size, NCCL_F64, nleft, stream.ptr)
    if nrght != MPI.PROC_NULL:
        nc.send(rght_send.data.ptr, rght_send.size, NCCL_F64, nrght, stream.ptr)
    nccl.groupEnd()
    stream.synchronize()

    if nrght != MPI.PROC_NULL:
        x[jlow:jhgh, nhalo+isize:isize+2*nhalo] = rght_recv
    if nleft != MPI.PROC_NULL:
        x[jlow:jhgh, 0:nhalo] = left_recv

    # --- top/bottom exchange (halo rows) ---
    if corners:
        width = 2*nhalo + isize
        cbeg = 0
    else:
        width = isize
        cbeg = nhalo

    bot_send = cp.ascontiguousarray(x[nhalo:2*nhalo, cbeg:cbeg+width]) if nbot != MPI.PROC_NULL else None
    top_send = cp.ascontiguousarray(x[jsize:nhalo+jsize, cbeg:cbeg+width]) if ntop != MPI.PROC_NULL else None
    bot_recv = cp.empty((nhalo, width), dtype=np.float64) if nbot != MPI.PROC_NULL else None
    top_recv = cp.empty((nhalo, width), dtype=np.float64) if ntop != MPI.PROC_NULL else None

    nccl.groupStart()
    if ntop != MPI.PROC_NULL:
        nc.recv(top_recv.data.ptr, top_recv.size, NCCL_F64, ntop, stream.ptr)
    if nbot != MPI.PROC_NULL:
        nc.recv(bot_recv.data.ptr, bot_recv.size, NCCL_F64, nbot, stream.ptr)
    if nbot != MPI.PROC_NULL:
        nc.send(bot_send.data.ptr, bot_send.size, NCCL_F64, nbot, stream.ptr)
    if ntop != MPI.PROC_NULL:
        nc.send(top_send.data.ptr, top_send.size, NCCL_F64, ntop, stream.ptr)
    nccl.groupEnd()
    stream.synchronize()

    if ntop != MPI.PROC_NULL:
        x[nhalo+jsize:jsize+2*nhalo, cbeg:cbeg+width] = top_recv
    if nbot != MPI.PROC_NULL:
        x[0:nhalo, cbeg:cbeg+width] = bot_recv

    ghostcell_time += time.time() - tstart_ghostcell


def create_netcdf_file(fname, jmax, imax, comm):

    ncid = Dataset(fname, 'w', format='NETCDF4', parallel=True, comm=comm)

    dimid_t = ncid.createDimension('time', None)  # None = unlimited dimension
    dimid_y = ncid.createDimension('y', jmax)
    dimid_x = ncid.createDimension('x', imax)

    varid = ncid.createVariable('u', 'f8', ('time', 'y', 'x'))  # f8 = double precision
    varid_xcoord = ncid.createVariable('xcoord', 'f8', ('x',))
    varid_ycoord = ncid.createVariable('ycoord', 'f8', ('y',))

    varid.description = "solution field"
    varid_xcoord.description = "x coordinate"
    varid_ycoord.description = "y coordinate"

    return ncid, varid, varid_xcoord, varid_ycoord


def write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, tstep, ncid, varid):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    xcoord = rank % nprocx
    ycoord = rank // nprocx

    ibegin = imax * xcoord // nprocx
    iend = imax * (xcoord + 1) // nprocx
    isize = iend - ibegin

    jbegin = jmax * ycoord // nprocy
    jend = jmax * (ycoord + 1) // nprocy
    jsize = jend - jbegin

    buf = x[nhalo:nhalo+jsize, nhalo:nhalo+isize][::-1, :]

    varid.set_collective(True)

    varid[tstep, jmax-jend:jmax-jbegin, ibegin:iend] = buf

    comm.Barrier()


def write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    xcoord = rank % nprocx
    ycoord = rank // nprocx

    ibegin = imax * xcoord // nprocx
    iend = imax * (xcoord + 1) // nprocx

    jbegin = jmax * ycoord // nprocy
    jend = jmax * (ycoord + 1) // nprocy

    varid_xcoord[ibegin:iend] = np.linspace(ibegin * Lx / imax,
                                            (iend - 1) * Lx / imax,
                                            iend - ibegin)

    varid_ycoord[jbegin:jend] = np.linspace(jbegin * Ly / jmax,
                                            (jend - 1) * Ly / jmax,
                                            jend - jbegin)

    comm.Barrier()


def close_netcdf(ncid):

    ncid.close()


def main():

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_time = 0.0

    if rank == 0:
        print("------> Initializing the Problem")

    # default values
    imax = 2000
    jmax = 2000
    nprocx = 0
    nprocy = 0
    nhalo = 2
    corners = 0
    do_timing = 0
    do_print = 0
    maxIter = 1000

    jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print = \
        parse_input_args(sys.argv, jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print)

    uid = nccl.get_unique_id() if rank == 0 else None
    uid = comm.bcast(uid, root=0)
    nc = nccl.NcclCommunicator(nprocs, uid, rank)
    stream = cp.cuda.Stream()

    stencil_time = 0.0
    tstart_total = time.time()

    xcoord = rank % nprocx
    ycoord = rank // nprocx

    nleft = rank - 1 if xcoord > 0 else MPI.PROC_NULL
    nrght = rank + 1 if xcoord < nprocx - 1 else MPI.PROC_NULL
    nbot = rank - nprocx if ycoord > 0 else MPI.PROC_NULL
    ntop = rank + nprocx if ycoord < nprocy - 1 else MPI.PROC_NULL

    ibegin = imax * xcoord // nprocx
    iend = imax * (xcoord + 1) // nprocx
    isize = iend - ibegin
    jbegin = jmax * ycoord // nprocy
    jend = jmax * (ycoord + 1) // nprocy
    jsize = jend - jbegin

    # physical domain dimensions (unit square)
    Lx = 1.0
    Ly = 1.0

    # soln init params
    sigma = 0.01 / ((Lx / imax) * (Ly / jmax))
    x_center = imax / 2.0
    y_center = jmax / 2.0

    # center of the Gaussian for initialization
    x0 = Lx / 2.0
    y0 = Ly / 2.0

    # allocate solution
    jsize_with_halos = jsize + 2*nhalo
    isize_with_halos = isize + 2*nhalo
    x = cp.zeros((jsize_with_halos, isize_with_halos), dtype=np.float64)
    xnew = cp.zeros((jsize_with_halos, isize_with_halos), dtype=np.float64)

    # init soln
    i_indices, j_indices = cp.meshgrid(cp.arange(isize), cp.arange(jsize))

    x_phys = i_indices + ibegin
    y_phys = j_indices + jbegin

    x[nhalo:jsize+nhalo, nhalo:isize+nhalo] = cp.exp(-0.5 * (((x_phys - x_center)**2 / (sigma**2)) +
                                      ((y_phys - y_center)**2 / (sigma**2))))

    # kernels launches are non-blocking for CuPy as in HIP
    cp.cuda.Device().synchronize()

    boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop)
    ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, nc, stream, do_timing)

    if do_print:
        ncid, varid, varid_xcoord, varid_ycoord = create_netcdf_file("solution.nc", jmax, imax, comm)
        x_cpu = cp.asnumpy(x)
        write_netcdf_soln(x_cpu, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid)
        write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord)

    if rank == 0:
        print("------> Advancing the Solution\n")

    for iter in range(maxIter):
        tstart_stencil = time.time()
        xnew[nhalo:nhalo+jsize, nhalo:nhalo+isize] = (
                                            x[nhalo:nhalo+jsize, nhalo:nhalo+isize] +
                                            x[nhalo:nhalo+jsize, nhalo-1:nhalo+isize-1] +
                                            x[nhalo:nhalo+jsize, nhalo+1:nhalo+isize+1] +
                                            x[nhalo-1:nhalo+jsize-1, nhalo:nhalo+isize] +
                                            x[nhalo+1:nhalo+jsize+1, nhalo:nhalo+isize]
                                                     )/5.0

        cp.cuda.Device().synchronize()
        # swap pointers
        x, xnew = xnew, x

        stencil_time += time.time() - tstart_stencil

        boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop)
        ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, nc, stream, do_timing)

        if do_print:
            if iter == maxIter - 1:
                x_cpu = cp.asnumpy(x)
                write_netcdf_soln(x_cpu, jmax, imax, nhalo, nprocy, nprocx, iter+1, ncid, varid)

    total_time += time.time() - tstart_total

    if rank == 0:
        print("------> Printing Timings")
        print(f"        Solution Advancement: {stencil_time:.6f}")
        print(f"        Boundary Condition Enforcement: {boundarycondition_time:.6f}")
        print(f"        Ghost Cell Update: {ghostcell_time:.6f}")
        print(f"        Total: {total_time:.6f}")


if __name__ == "__main__":
    main()
