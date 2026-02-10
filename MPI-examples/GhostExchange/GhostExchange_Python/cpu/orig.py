from mpi4py import MPI
from netCDF4 import Dataset
import numpy as np
import time
import argparse
import sys

boundarycondition_time = 0.0
ghostcell_time=0.0

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


def ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing):
    
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
    bufcount = jnum * nhalo

    xbuf_left_send = np.zeros(bufcount, dtype=np.float64)
    xbuf_rght_send = np.zeros(bufcount, dtype=np.float64)
    xbuf_rght_recv = np.zeros(bufcount, dtype=np.float64)
    xbuf_left_recv = np.zeros(bufcount, dtype=np.float64)

    if nleft != MPI.PROC_NULL:
        xbuf_left_send = x[jlow:jhgh, nhalo:2*nhalo].flatten()
    if nrght != MPI.PROC_NULL:
        xbuf_rght_send = x[jlow:jhgh, isize:nhalo+isize].flatten()

    requests = []

    if nrght != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Irecv(xbuf_rght_recv, source=nrght, tag=1001))
    if nleft != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Irecv(xbuf_left_recv, source=nleft, tag=1002))

    if nleft != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Isend(xbuf_left_send, dest=nleft, tag=1001))
    if nrght != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Isend(xbuf_rght_send, dest=nrght, tag=1002))

    MPI.Request.Waitall(requests)

    if nrght != MPI.PROC_NULL:
        x[jlow:jhgh, nhalo+isize:isize+2*nhalo] = xbuf_rght_recv.reshape(jnum, nhalo)
    if nleft != MPI.PROC_NULL:
        x[jlow:jhgh, 0:nhalo] = xbuf_left_recv.reshape(jnum, nhalo)

    requests = []

    if corners:
        bufcount = nhalo * (isize + 2*nhalo)

        if ntop != MPI.PROC_NULL:
            recv_buf_top = np.zeros((nhalo, isize + 2*nhalo), dtype=np.float64)
            requests.append(MPI.COMM_WORLD.Irecv(recv_buf_top, source=ntop, tag=2001))
        if nbot != MPI.PROC_NULL:
            send_buf_bot = x[nhalo:2*nhalo, 0:2*nhalo+isize].copy()
            requests.append(MPI.COMM_WORLD.Isend(send_buf_bot, dest=nbot, tag=2001))
        if nbot != MPI.PROC_NULL:
            recv_buf_bot = np.zeros((nhalo, isize + 2*nhalo), dtype=np.float64)
            requests.append(MPI.COMM_WORLD.Irecv(recv_buf_bot, source=nbot, tag=2002))
        if ntop != MPI.PROC_NULL:
            send_buf_top = x[jsize:nhalo+jsize, 0:2*nhalo+isize].copy()
            requests.append(MPI.COMM_WORLD.Isend(send_buf_top, dest=ntop, tag=2002))

        MPI.Request.Waitall(requests)

        if ntop != MPI.PROC_NULL:
            x[nhalo+jsize:jsize+2*nhalo, 0:2*nhalo+isize] = recv_buf_top
        if nbot != MPI.PROC_NULL:
            x[0:nhalo, 0:2*nhalo+isize] = recv_buf_bot

    else:
        for j in range(nhalo):
            if ntop != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Irecv(
                    x[nhalo+jsize+j, nhalo:nhalo+isize],
                    source=ntop,
                    tag=3000+j*2
                ))
            if nbot != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Isend(
                    x[nhalo+j, nhalo:nhalo+isize].copy(),
                    dest=nbot,
                    tag=3000+j*2
                ))
            if nbot != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Irecv(
                    x[j, nhalo:nhalo+isize],
                    source=nbot,
                    tag=3001+j*2
                ))
            if ntop != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Isend(
                    x[jsize+j, nhalo:nhalo+isize].copy(),
                    dest=ntop,
                    tag=3001+j*2
                ))

        MPI.Request.Waitall(requests)   

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
    jsize_with_halos=jsize + 2*nhalo
    isize_with_halos=isize + 2*nhalo
    x = np.zeros((jsize_with_halos, isize_with_halos), dtype=np.float64)
    xnew = np.zeros((jsize_with_halos, isize_with_halos), dtype=np.float64)

    # init soln
    i_indices, j_indices = np.meshgrid(np.arange(isize), np.arange(jsize))

    x_phys = i_indices + ibegin
    y_phys = j_indices + jbegin

    x[nhalo:jsize+nhalo, nhalo:isize+nhalo] = np.exp(-0.5 * (((x_phys - x_center)**2 / (sigma**2)) +
                                      ((y_phys - y_center)**2 / (sigma**2))))
    

    boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
    ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);


    if do_print:
       ncid, varid, varid_xcoord, varid_ycoord = create_netcdf_file("solution.nc", jmax, imax, comm)
       write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid);
       write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord);

    if rank == 0:
         print("------> Advancing the Solution\n");

    for iter in range(maxIter):
       tstart_stencil = time.time()
       xnew[nhalo:nhalo+jsize, nhalo:nhalo+isize] = (
                                           x[nhalo:nhalo+jsize, nhalo:nhalo+isize] +
                                           x[nhalo:nhalo+jsize, nhalo-1:nhalo+isize-1] +
                                           x[nhalo:nhalo+jsize, nhalo+1:nhalo+isize+1] +
                                           x[nhalo-1:nhalo+jsize-1, nhalo:nhalo+isize] +
                                           x[nhalo+1:nhalo+jsize+1, nhalo:nhalo+isize]
                                                    )/5.0


       # swap pointers
       x, xnew = xnew, x

       stencil_time += time.time() - tstart_stencil

       boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
       ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

       if do_print:
           if iter == maxIter - 1:
              write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, iter+1, ncid, varid);

       total_time += time.time() - tstart_total

    if rank == 0:
       print("------> Printing Timings")
       print(f"        Solution Advancement: {stencil_time:.6f}")
       print(f"        Boundary Condition Enforcement: {boundarycondition_time:.6f}")
       print(f"        Ghost Cell Update: {ghostcell_time:.6f}")
       print(f"        Total: {total_time:.6f}")

if __name__ == "__main__":
    main()

