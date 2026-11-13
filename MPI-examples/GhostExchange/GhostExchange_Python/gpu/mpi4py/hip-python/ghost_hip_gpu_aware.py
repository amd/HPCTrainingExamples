#!/usr/bin/env python3

from mpi4py import MPI
from netCDF4 import Dataset
import numpy as np
import ctypes
import time
import sys
from hip import hip, hiprtc

boundarycondition_time = 0.0
ghostcell_time = 0.0


class _DeviceCAI:
    # __cuda_array_interface__ view of a device pointer, so mpi4py passes it to a GPU-aware MPI
    def __init__(self, ptr, count):
        self.__cuda_array_interface__ = {
            'shape': (int(count),),
            'typestr': '<f8',
            'data': (int(ptr), False),
            'version': 3,
            'strides': None,
        }


def hip_check(call_result):
    # hip-python returns (error, results...); some calls return just the error
    if isinstance(call_result, tuple):
        err, result = call_result[0], call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        err, result = call_result, None
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(str(err))
    return result


# HIP kernels of the C++ HIP version (same names; nhalo and ld added).
# X(j, i) is x[j][i] of the C++ code: interior cell (j, i), halo at negative indices.
KERNEL_SOURCE = b"""
#define X(j, i)    x[((j) + nhalo) * ld + ((i) + nhalo)]
#define XNEW(j, i) xnew[((j) + nhalo) * ld + ((i) + nhalo)]

extern "C" __global__
void init_core(double* x, int nhalo, int jsize, int isize, double val, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y - nhalo;
    if (tidy < jsize + nhalo && tidx < isize + nhalo) {
        X(tidy, tidx) = val;
    }
}

extern "C" __global__
void init_core2(double* x, int jsize, int isize, int jbegin, int ibegin, double sigma,
                double x_center, double y_center, int nhalo, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < isize) {
        double x_phys = tidx + ibegin;
        double y_phys = tidy + jbegin;
        X(tidy, tidx) = exp(-0.5 * ( ((x_phys - x_center) * (x_phys - x_center) / (sigma * sigma)
                        + (y_phys - y_center) * (y_phys - y_center) / (sigma * sigma)) ));
    }
}

extern "C" __global__
void blur(const double* x, double* xnew, int jsize, int isize, int nhalo, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < isize) {
        XNEW(tidy, tidx) = ( X(tidy, tidx) + X(tidy, tidx-1) + X(tidy, tidx+1) + X(tidy-1, tidx) + X(tidy+1, tidx) )/5.0;
    }
}

extern "C" __global__
void enforce_bcs_left(double* x, int nhalo, int jsize, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < 0) {
        X(tidy, tidx) = X(tidy, 0);
    }
}

extern "C" __global__
void enforce_bcs_rght(double* x, int nhalo, int jsize, int isize, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < nhalo) {
        X(tidy, isize + tidx) = X(tidy, isize - 1);
    }
}

extern "C" __global__
void enforce_bcs_bot(double* x, int nhalo, int isize, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y - nhalo;
    if (tidy < 0 && tidx < isize + nhalo) {
        X(tidy, tidx) = X(0, tidx);
    }
}

extern "C" __global__
void enforce_bcs_top(double* x, int nhalo, int jsize, int isize, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < nhalo && tidx < isize + nhalo) {
        X(jsize + tidy, tidx) = X(jsize - 1, tidx);
    }
}

extern "C" __global__
void ghost_left_send(double* xbuf_left_send, const double* x, int jlow, int jhgh, int nhalo, int bufcount, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xbuf_left_send[offset] = X(tidy, tidx);
    }
}

extern "C" __global__
void ghost_rght_send(double* xbuf_rght_send, const double* x, int jlow, int jhgh, int isize, int nhalo, int bufcount, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xbuf_rght_send[offset] = X(tidy, isize - nhalo + tidx);
    }
}

extern "C" __global__
void ghost_rght_recv(const double* xbuf_rght_recv, double* x, int jlow, int jhgh, int isize, int nhalo, int bufcount, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        X(tidy, isize + tidx) = xbuf_rght_recv[offset];
    }
}

extern "C" __global__
void ghost_left_recv(const double* xbuf_left_recv, double* x, int jlow, int jhgh, int nhalo, int bufcount, int ld)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        X(tidy, tidx - nhalo) = xbuf_left_recv[offset];
    }
}
"""

KERNEL_NAMES = ("init_core", "init_core2", "blur",
                "enforce_bcs_left", "enforce_bcs_rght", "enforce_bcs_bot", "enforce_bcs_top",
                "ghost_left_send", "ghost_rght_send", "ghost_rght_recv", "ghost_left_recv")


def compile_kernels():
    # compile the kernels for this GPU with HIPRTC
    prog = hip_check(hiprtc.hiprtcCreateProgram(KERNEL_SOURCE, b"ghost_kernels", 0, [], []))
    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    cflags = [b"--offload-arch=" + props.gcnArchName]
    err = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    err = err[0] if isinstance(err, tuple) else err
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        raise RuntimeError("HIPRTC compile failed:\n" + log.decode())
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernels = {name: hip_check(hip.hipModuleGetFunction(module, name.encode())) for name in KERNEL_NAMES}
    return module, kernels


def xptr(x_d, j, i, nhalo, ld):
    # device address of x[j][i]
    return int(x_d) + ((j + nhalo) * ld + (i + nhalo)) * 8


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------
def parse_input_args(argv, jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    i = 1
    while i < len(argv):
        if argv[i] == '-c':
            corners = 1; i += 1
        elif argv[i] == '-t':
            do_timing = 1; i += 1
        elif argv[i] == '-p':
            do_print = 1; i += 1
        elif argv[i] == '-h':
            nhalo = int(argv[i + 1]); i += 2
        elif argv[i] == '-I':
            maxIter = int(argv[i + 1]); i += 2
        elif argv[i] == '-i':
            imax = int(argv[i + 1]); i += 2
        elif argv[i] == '-j':
            jmax = int(argv[i + 1]); i += 2
        elif argv[i] == '-x':
            nprocx = int(argv[i + 1]); i += 2
        elif argv[i] == '-y':
            nprocy = int(argv[i + 1]); i += 2
        else:
            if rank == 0:
                print(f"Unknown option {argv[i]}", file=sys.stderr)
            sys.exit(1)
    return jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print


# ----------------------------------------------------------------------
# boundary conditions at the physical borders
# ----------------------------------------------------------------------
def boundarycondition_update(K, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop):
    global boundarycondition_time
    t = time.time()

    gridVrt = ((nhalo + 3) // 4, (jsize + 63) // 64, 1)
    blockVrt = (4, 64, 1)
    gridHor = ((isize + 2 * nhalo + 63) // 64, (nhalo + 3) // 4, 1)
    blockHor = (64, 4, 1)

    if nleft == MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["enforce_bcs_left"], *gridVrt, *blockVrt,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, ctypes.c_int(nhalo), ctypes.c_int(jsize), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    if nrght == MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["enforce_bcs_rght"], *gridVrt, *blockVrt,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, ctypes.c_int(nhalo), ctypes.c_int(jsize), ctypes.c_int(isize), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    if nbot == MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["enforce_bcs_bot"], *gridHor, *blockHor,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, ctypes.c_int(nhalo), ctypes.c_int(isize), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    if ntop == MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["enforce_bcs_top"], *gridHor, *blockHor,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, ctypes.c_int(nhalo), ctypes.c_int(jsize), ctypes.c_int(isize), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    boundarycondition_time += time.time() - t


# ----------------------------------------------------------------------
# ghost-cell exchange (GPU-aware MPI)
# ----------------------------------------------------------------------
def ghostcell_update(K, x_d, buf, nhalo, corners, jsize, isize, ld, nleft, nrght, nbot, ntop):
    global ghostcell_time
    t = time.time()

    jlow, jhgh = 0, jsize
    if corners:
        if nbot == MPI.PROC_NULL:
            jlow = -nhalo
        if ntop == MPI.PROC_NULL:
            jhgh = jsize + nhalo
    jnum = jhgh - jlow
    bufcount = jnum * nhalo

    # left/right: pack the halo columns, exchange, unpack
    gridGhst = ((nhalo + 3) // 4, (jnum + 63) // 64, 1)
    blockGhst = (4, 64, 1)

    if nleft != MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["ghost_left_send"], *gridGhst, *blockGhst,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(buf['xbuf_left_send'], x_d, ctypes.c_int(jlow), ctypes.c_int(jhgh),
                         ctypes.c_int(nhalo), ctypes.c_int(bufcount), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    if nrght != MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["ghost_rght_send"], *gridGhst, *blockGhst,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(buf['xbuf_rght_send'], x_d, ctypes.c_int(jlow), ctypes.c_int(jhgh),
                         ctypes.c_int(isize), ctypes.c_int(nhalo), ctypes.c_int(bufcount), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    requests = []
    if nrght != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(buf['xbuf_rght_recv'], bufcount), source=nrght, tag=1001))
    if nleft != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(buf['xbuf_left_send'], bufcount), dest=nleft, tag=1001))
    if nleft != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(buf['xbuf_left_recv'], bufcount), source=nleft, tag=1002))
    if nrght != MPI.PROC_NULL:
        requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(buf['xbuf_rght_send'], bufcount), dest=nrght, tag=1002))
    MPI.Request.Waitall(requests)
    hip_check(hip.hipDeviceSynchronize())

    if nrght != MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["ghost_rght_recv"], *gridGhst, *blockGhst,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(buf['xbuf_rght_recv'], x_d, ctypes.c_int(jlow), ctypes.c_int(jhgh),
                         ctypes.c_int(isize), ctypes.c_int(nhalo), ctypes.c_int(bufcount), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    if nleft != MPI.PROC_NULL:
        hip_check(hip.hipModuleLaunchKernel(K["ghost_left_recv"], *gridGhst, *blockGhst,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(buf['xbuf_left_recv'], x_d, ctypes.c_int(jlow), ctypes.c_int(jhgh),
                         ctypes.c_int(nhalo), ctypes.c_int(bufcount), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    # up/down: the halo rows are contiguous, so no packing
    if corners:
        bufcount = nhalo * (isize + 2 * nhalo)
        requests = []
        if ntop != MPI.PROC_NULL:
            requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(xptr(x_d, jsize, -nhalo, nhalo, ld), bufcount), source=ntop, tag=1001))
        if nbot != MPI.PROC_NULL:
            requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(xptr(x_d, 0, -nhalo, nhalo, ld), bufcount), dest=nbot, tag=1001))
        if nbot != MPI.PROC_NULL:
            requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(xptr(x_d, -nhalo, -nhalo, nhalo, ld), bufcount), source=nbot, tag=1002))
        if ntop != MPI.PROC_NULL:
            requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(xptr(x_d, jsize - nhalo, -nhalo, nhalo, ld), bufcount), dest=ntop, tag=1002))
        MPI.Request.Waitall(requests)
    else:
        requests = []
        for j in range(nhalo):
            if ntop != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(xptr(x_d, jsize + j, 0, nhalo, ld), isize), source=ntop, tag=1001 + j * 2))
            if nbot != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(xptr(x_d, j, 0, nhalo, ld), isize), dest=nbot, tag=1001 + j * 2))
            if nbot != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Irecv(_DeviceCAI(xptr(x_d, -nhalo + j, 0, nhalo, ld), isize), source=nbot, tag=1002 + j * 2))
            if ntop != MPI.PROC_NULL:
                requests.append(MPI.COMM_WORLD.Isend(_DeviceCAI(xptr(x_d, jsize - nhalo + j, 0, nhalo, ld), isize), dest=ntop, tag=1002 + j * 2))
        MPI.Request.Waitall(requests)
    hip_check(hip.hipDeviceSynchronize())

    ghostcell_time += time.time() - t


# ----------------------------------------------------------------------
# netCDF output
# ----------------------------------------------------------------------
def create_netcdf_file(fname, jmax, imax, comm):
    ncid = Dataset(fname, 'w', format='NETCDF4', parallel=True, comm=comm)
    ncid.createDimension('time', None)
    ncid.createDimension('y', jmax)
    ncid.createDimension('x', imax)
    varid = ncid.createVariable('u', 'f8', ('time', 'y', 'x'))
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
    buf = x[nhalo:nhalo + jsize, nhalo:nhalo + isize][::-1, :]
    varid.set_collective(True)
    varid[tstep, jmax - jend:jmax - jbegin, ibegin:iend] = buf
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
    varid_xcoord[ibegin:iend] = np.linspace(ibegin * Lx / imax, (iend - 1) * Lx / imax, iend - ibegin)
    varid_ycoord[jbegin:jend] = np.linspace(jbegin * Ly / jmax, (jend - 1) * Ly / jmax, jend - jbegin)
    comm.Barrier()


def close_netcdf(ncid):
    ncid.close()


# ----------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_time = 0.0

    if rank == 0:
        print("------> Initializing the Problem")

    imax = 2000; jmax = 2000
    nprocx = 0; nprocy = 0
    nhalo = 2; corners = 0
    do_timing = 0; do_print = 0
    maxIter = 1000

    jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print = \
        parse_input_args(sys.argv, jmax, imax, nprocy, nprocx, nhalo, corners,
                         maxIter, do_timing, do_print)

    module, K = compile_kernels()

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

    Lx = 1.0; Ly = 1.0
    sigma = 0.01 / ((Lx / imax) * (Ly / jmax))
    x_center = imax / 2.0
    y_center = jmax / 2.0

    # solution arrays, with nhalo ghost layers on every side
    ld = isize + 2 * nhalo
    nbytes = (jsize + 2 * nhalo) * ld * 8
    x_d = hip_check(hip.hipMalloc(nbytes))
    xnew_d = hip_check(hip.hipMalloc(nbytes))

    # halo message buffers
    jlow = -nhalo if (corners and nbot == MPI.PROC_NULL) else 0
    jhgh = jsize + nhalo if (corners and ntop == MPI.PROC_NULL) else jsize
    bufcount = (jhgh - jlow) * nhalo
    buf = {}
    for name in ('xbuf_left_send', 'xbuf_rght_send', 'xbuf_rght_recv', 'xbuf_left_recv'):
        buf[name] = hip_check(hip.hipMalloc(bufcount * 8))

    if not corners:  # need to initialize when not doing corners so there is no uninitialized memory
        grid = ((isize + 2 * nhalo + 63) // 64, (jsize + 2 * nhalo + 3) // 4, 1)
        block = (64, 4, 1)
        hip_check(hip.hipModuleLaunchKernel(K["init_core"], *grid, *block,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, ctypes.c_int(nhalo), ctypes.c_int(jsize), ctypes.c_int(isize),
                         ctypes.c_double(0.0), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())

    grid = ((isize + 63) // 64, (jsize + 3) // 4, 1)
    block = (64, 4, 1)
    hip_check(hip.hipModuleLaunchKernel(K["init_core2"], *grid, *block,
              sharedMemBytes=0, stream=None, kernelParams=None,
              extra=(x_d, ctypes.c_int(jsize), ctypes.c_int(isize), ctypes.c_int(jbegin), ctypes.c_int(ibegin),
                     ctypes.c_double(sigma), ctypes.c_double(x_center), ctypes.c_double(y_center),
                     ctypes.c_int(nhalo), ctypes.c_int(ld))))
    hip_check(hip.hipDeviceSynchronize())

    boundarycondition_update(K, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop)
    ghostcell_update(K, x_d, buf, nhalo, corners, jsize, isize, ld, nleft, nrght, nbot, ntop)

    if do_print:
        ncid, varid, varid_xcoord, varid_ycoord = create_netcdf_file("solution.nc", jmax, imax, comm)
        x_h = np.zeros((jsize + 2 * nhalo, ld), dtype=np.float64)
        hip_check(hip.hipMemcpy(x_h, x_d, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        write_netcdf_soln(x_h, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid)
        write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord)

    if rank == 0:
        print("------> Advancing the Solution\n")

    for it in range(maxIter):
        t = time.time()
        hip_check(hip.hipModuleLaunchKernel(K["blur"], *grid, *block,
                  sharedMemBytes=0, stream=None, kernelParams=None,
                  extra=(x_d, xnew_d, ctypes.c_int(jsize), ctypes.c_int(isize), ctypes.c_int(nhalo), ctypes.c_int(ld))))
        hip_check(hip.hipDeviceSynchronize())
        x_d, xnew_d = xnew_d, x_d
        stencil_time += time.time() - t

        boundarycondition_update(K, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop)
        ghostcell_update(K, x_d, buf, nhalo, corners, jsize, isize, ld, nleft, nrght, nbot, ntop)

        if do_print and it == maxIter - 1:
            x_h = np.zeros((jsize + 2 * nhalo, ld), dtype=np.float64)
            hip_check(hip.hipMemcpy(x_h, x_d, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
            write_netcdf_soln(x_h, jmax, imax, nhalo, nprocy, nprocx, it + 1, ncid, varid)

    total_time += time.time() - tstart_total

    if rank == 0:
        print("------> Printing Timings")
        print(f"        Solution Advancement: {stencil_time:.6f}")
        print(f"        Boundary Condition Enforcement: {boundarycondition_time:.6f}")
        print(f"        Ghost Cell Update: {ghostcell_time:.6f}")
        print(f"        Total: {total_time:.6f}")


if __name__ == "__main__":
    main()
