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
    # Minimal 1-D float64 __cuda_array_interface__ view over a device pointer,
    # so mpi4py hands the buffer straight to a GPU-aware MPI (no host copy).
    def __init__(self, ptr, count):
        self.__cuda_array_interface__ = {
            'shape': (int(count),),
            'typestr': '<f8',
            'data': (int(ptr), False),
            'version': 3,
            'strides': None,
        }


def cai(dev_buf, count):
    # Wrap a hip-python device allocation (DeviceArray) as a CAI buffer of
    # exactly `count` float64 elements. int(dev_buf) is the device address.
    return _DeviceCAI(int(dev_buf), count)


# ----------------------------------------------------------------------
# HIP helpers
# ----------------------------------------------------------------------
def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(str(err))
    return result


KERNEL_SOURCE = b"""
extern "C" __global__
void init_gaussian(double* x, int jsize, int isize, int nhalo, int ld,
                   int ibegin, int jbegin,
                   double x_center, double y_center, double sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 0..isize-1
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // 0..jsize-1
    if (i < isize && j < jsize) {
        double xp = (double)(i + ibegin);
        double yp = (double)(j + jbegin);
        double e  = ((xp - x_center) * (xp - x_center)) / (sigma * sigma)
                  + ((yp - y_center) * (yp - y_center)) / (sigma * sigma);
        x[(j + nhalo) * ld + (i + nhalo)] = exp(-0.5 * e);
    }
}

extern "C" __global__
void stencil(const double* x, double* xnew, int jsize, int isize, int nhalo, int ld) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < isize && j < jsize) {
        int r = j + nhalo;
        int c = i + nhalo;
        xnew[r * ld + c] = ( x[r * ld + c]
                           + x[r * ld + (c - 1)]
                           + x[r * ld + (c + 1)]
                           + x[(r - 1) * ld + c]
                           + x[(r + 1) * ld + c] ) / 5.0;
    }
}

// pack a [rows x cols] block starting at (r0,c0) into a contiguous buffer
extern "C" __global__
void pack_block(const double* x, double* buf, int r0, int c0,
                int rows, int cols, int ld) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        buf[idx] = x[(r0 + r) * ld + (c0 + c)];
    }
}

// scatter a contiguous buffer into a [rows x cols] block at (r0,c0)
extern "C" __global__
void unpack_block(double* x, const double* buf, int r0, int c0,
                  int rows, int cols, int ld) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        x[(r0 + r) * ld + (c0 + c)] = buf[idx];
    }
}

// device-to-device copy of a [rows x cols] block (src -> dst), same array
extern "C" __global__
void copy_block(double* x, int dr0, int dc0, int sr0, int sc0,
                int rows, int cols, int ld) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        x[(dr0 + r) * ld + (dc0 + c)] = x[(sr0 + r) * ld + (sc0 + c)];
    }
}
"""


class Hip:
    """Compiled kernels + convenience launch wrappers."""

    def __init__(self):
        prog = hip_check(hiprtc.hiprtcCreateProgram(KERNEL_SOURCE, b"ghost_kernels", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch=" + arch]
        (err,) = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError("HIPRTC compile failed:\n" + log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.k = {name: hip_check(hip.hipModuleGetFunction(self.module, name.encode()))
                  for name in ("init_gaussian", "stencil", "pack_block",
                               "unpack_block", "copy_block")}

    def launch1d(self, name, total, args):
        block = 256
        grid = (total + block - 1) // block
        hip_check(hip.hipModuleLaunchKernel(
            self.k[name], grid, 1, 1, block, 1, 1,
            sharedMemBytes=0, stream=None, kernelParams=None, extra=args))

    def launch2d(self, name, nx, ny, args):
        bx, by = 16, 16
        gx = (nx + bx - 1) // bx
        gy = (ny + by - 1) // by
        hip_check(hip.hipModuleLaunchKernel(
            self.k[name], gx, gy, 1, bx, by, 1,
            sharedMemBytes=0, stream=None, kernelParams=None, extra=args))


def dmalloc(nbytes):
    d = hip_check(hip.hipMalloc(nbytes))
    hip_check(hip.hipMemset(d, 0, nbytes))
    return d


# ----------------------------------------------------------------------
# argument parsing (identical to the CuPy version)
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
# boundary conditions (replicate edge into the halo for physical borders)
# ----------------------------------------------------------------------
def boundarycondition_update(H, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop):
    global boundarycondition_time
    t = time.time()

    if nleft == MPI.PROC_NULL:
        for k in range(nhalo):
            H.launch1d("copy_block", jsize,
                       (x_d, ctypes.c_int(nhalo), ctypes.c_int(k),
                        ctypes.c_int(nhalo), ctypes.c_int(nhalo),
                        ctypes.c_int(jsize), ctypes.c_int(1), ctypes.c_int(ld)))
    if nrght == MPI.PROC_NULL:
        for k in range(nhalo):
            H.launch1d("copy_block", jsize,
                       (x_d, ctypes.c_int(nhalo), ctypes.c_int(isize + nhalo + k),
                        ctypes.c_int(nhalo), ctypes.c_int(isize + nhalo - 1),
                        ctypes.c_int(jsize), ctypes.c_int(1), ctypes.c_int(ld)))
    width = isize + 2 * nhalo
    if nbot == MPI.PROC_NULL:
        for k in range(nhalo):
            H.launch1d("copy_block", width,
                       (x_d, ctypes.c_int(k), ctypes.c_int(0),
                        ctypes.c_int(nhalo), ctypes.c_int(0),
                        ctypes.c_int(1), ctypes.c_int(width), ctypes.c_int(ld)))
    if ntop == MPI.PROC_NULL:
        for k in range(nhalo):
            H.launch1d("copy_block", width,
                       (x_d, ctypes.c_int(jsize + nhalo + k), ctypes.c_int(0),
                        ctypes.c_int(jsize + nhalo - 1), ctypes.c_int(0),
                        ctypes.c_int(1), ctypes.c_int(width), ctypes.c_int(ld)))

    hip_check(hip.hipDeviceSynchronize())
    boundarycondition_time += time.time() - t


# ----------------------------------------------------------------------
# ghost-cell exchange (GPU-aware MPI). Mirrors the CuPy corners path.
# ----------------------------------------------------------------------
def ghostcell_update(H, x_d, s, nhalo, corners, jsize, isize, ld,
                     nleft, nrght, nbot, ntop):
    global ghostcell_time
    t = time.time()

    jlow = nhalo
    jhgh = nhalo + jsize
    if corners:
        if nbot == MPI.PROC_NULL:
            jlow = 0
        if ntop == MPI.PROC_NULL:
            jhgh = nhalo + jsize + nhalo
    jnum = jhgh - jlow
    col_count = jnum * nhalo

    # ----- left/right columns -----
    if nleft != MPI.PROC_NULL:
        H.launch1d("pack_block", col_count,
                   (x_d, s['cl_send_d'], ctypes.c_int(jlow), ctypes.c_int(nhalo),
                    ctypes.c_int(jnum), ctypes.c_int(nhalo), ctypes.c_int(ld)))
    if nrght != MPI.PROC_NULL:
        H.launch1d("pack_block", col_count,
                   (x_d, s['cr_send_d'], ctypes.c_int(jlow), ctypes.c_int(isize),
                    ctypes.c_int(jnum), ctypes.c_int(nhalo), ctypes.c_int(ld)))
    # the packed send buffers must be complete before a GPU-aware MPI reads them
    hip_check(hip.hipDeviceSynchronize())

    reqs = []
    if nrght != MPI.PROC_NULL:
        reqs.append(MPI.COMM_WORLD.Irecv(cai(s['cr_recv_d'], col_count), source=nrght, tag=1001))
    if nleft != MPI.PROC_NULL:
        reqs.append(MPI.COMM_WORLD.Irecv(cai(s['cl_recv_d'], col_count), source=nleft, tag=1002))
    if nleft != MPI.PROC_NULL:
        reqs.append(MPI.COMM_WORLD.Isend(cai(s['cl_send_d'], col_count), dest=nleft, tag=1001))
    if nrght != MPI.PROC_NULL:
        reqs.append(MPI.COMM_WORLD.Isend(cai(s['cr_send_d'], col_count), dest=nrght, tag=1002))
    MPI.Request.Waitall(reqs)
    # received data is now on the device; make sure it has landed before unpack
    hip_check(hip.hipDeviceSynchronize())

    if nrght != MPI.PROC_NULL:
        H.launch1d("unpack_block", col_count,
                   (x_d, s['cr_recv_d'], ctypes.c_int(jlow), ctypes.c_int(nhalo + isize),
                    ctypes.c_int(jnum), ctypes.c_int(nhalo), ctypes.c_int(ld)))
    if nleft != MPI.PROC_NULL:
        H.launch1d("unpack_block", col_count,
                   (x_d, s['cl_recv_d'], ctypes.c_int(jlow), ctypes.c_int(0),
                    ctypes.c_int(jnum), ctypes.c_int(nhalo), ctypes.c_int(ld)))
    hip_check(hip.hipDeviceSynchronize())

    # ----- top/bottom rows (corners path) -----
    if corners:
        width = isize + 2 * nhalo
        row_count = nhalo * width

        if nbot != MPI.PROC_NULL:
            H.launch1d("pack_block", row_count,
                       (x_d, s['rb_send_d'], ctypes.c_int(nhalo), ctypes.c_int(0),
                        ctypes.c_int(nhalo), ctypes.c_int(width), ctypes.c_int(ld)))
        if ntop != MPI.PROC_NULL:
            H.launch1d("pack_block", row_count,
                       (x_d, s['rt_send_d'], ctypes.c_int(jsize), ctypes.c_int(0),
                        ctypes.c_int(nhalo), ctypes.c_int(width), ctypes.c_int(ld)))
        hip_check(hip.hipDeviceSynchronize())

        reqs = []
        if ntop != MPI.PROC_NULL:
            reqs.append(MPI.COMM_WORLD.Irecv(cai(s['rt_recv_d'], row_count), source=ntop, tag=2001))
        if nbot != MPI.PROC_NULL:
            reqs.append(MPI.COMM_WORLD.Isend(cai(s['rb_send_d'], row_count), dest=nbot, tag=2001))
        if nbot != MPI.PROC_NULL:
            reqs.append(MPI.COMM_WORLD.Irecv(cai(s['rb_recv_d'], row_count), source=nbot, tag=2002))
        if ntop != MPI.PROC_NULL:
            reqs.append(MPI.COMM_WORLD.Isend(cai(s['rt_send_d'], row_count), dest=ntop, tag=2002))
        MPI.Request.Waitall(reqs)
        hip_check(hip.hipDeviceSynchronize())

        if ntop != MPI.PROC_NULL:
            H.launch1d("unpack_block", row_count,
                       (x_d, s['rt_recv_d'], ctypes.c_int(nhalo + jsize), ctypes.c_int(0),
                        ctypes.c_int(nhalo), ctypes.c_int(width), ctypes.c_int(ld)))
        if nbot != MPI.PROC_NULL:
            H.launch1d("unpack_block", row_count,
                       (x_d, s['rb_recv_d'], ctypes.c_int(0), ctypes.c_int(0),
                        ctypes.c_int(nhalo), ctypes.c_int(width), ctypes.c_int(ld)))
        hip_check(hip.hipDeviceSynchronize())

    ghostcell_time += time.time() - t


# ----------------------------------------------------------------------
# netCDF output (identical to the CuPy version; operates on host arrays)
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

    # GPU selection is left to the launcher: bind one GPU per rank externally
    # (srun --gpu-bind or an affinity wrapper exposing a single GPU per rank).
    # The app uses the default device, which is that rank's bound GPU.
    H = Hip()

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

    jsize_wh = jsize + 2 * nhalo
    isize_wh = isize + 2 * nhalo
    ld = isize_wh
    nbytes = jsize_wh * isize_wh * 8

    x_d = dmalloc(nbytes)
    xnew_d = dmalloc(nbytes)

    # pre-allocated halo scratch (device + host), sized to the max message
    jlow = 0 if (corners and nbot == MPI.PROC_NULL) else nhalo
    jhgh = (nhalo + jsize + nhalo) if (corners and ntop == MPI.PROC_NULL) else (nhalo + jsize)
    jnum_max = jhgh - jlow
    col_max = jnum_max * nhalo
    row_max = nhalo * (isize + 2 * nhalo)
    s = {}
    for key in ('cl_send', 'cr_send', 'cl_recv', 'cr_recv'):
        s[key + '_d'] = dmalloc(col_max * 8)
    for key in ('rb_send', 'rt_send', 'rb_recv', 'rt_recv'):
        s[key + '_d'] = dmalloc(row_max * 8)

    # init Gaussian on device
    H.launch2d("init_gaussian", isize, jsize,
               (x_d, ctypes.c_int(jsize), ctypes.c_int(isize), ctypes.c_int(nhalo),
                ctypes.c_int(ld), ctypes.c_int(ibegin), ctypes.c_int(jbegin),
                ctypes.c_double(x_center), ctypes.c_double(y_center), ctypes.c_double(sigma)))
    hip_check(hip.hipDeviceSynchronize())

    boundarycondition_update(H, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop)
    ghostcell_update(H, x_d, s, nhalo, corners, jsize, isize, ld, nleft, nrght, nbot, ntop)

    if do_print:
        ncid, varid, varid_xcoord, varid_ycoord = create_netcdf_file("solution.nc", jmax, imax, comm)
        x_h = np.zeros((jsize_wh, isize_wh), dtype=np.float64)
        hip_check(hip.hipMemcpy(x_h, x_d, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        write_netcdf_soln(x_h, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid)
        write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord)

    if rank == 0:
        print("------> Advancing the Solution\n")

    for it in range(maxIter):
        t = time.time()
        H.launch2d("stencil", isize, jsize,
                   (x_d, xnew_d, ctypes.c_int(jsize), ctypes.c_int(isize),
                    ctypes.c_int(nhalo), ctypes.c_int(ld)))
        hip_check(hip.hipDeviceSynchronize())
        x_d, xnew_d = xnew_d, x_d
        stencil_time += time.time() - t

        boundarycondition_update(H, x_d, nhalo, jsize, isize, ld, nleft, nrght, nbot, ntop)
        ghostcell_update(H, x_d, s, nhalo, corners, jsize, isize, ld, nleft, nrght, nbot, ntop)

        if do_print and it == maxIter - 1:
            x_h = np.zeros((jsize_wh, isize_wh), dtype=np.float64)
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

