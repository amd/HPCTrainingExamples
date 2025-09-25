
# HIP-Python

The first test is to check that the hip-python environment is set up correctly.

```
module load rocm hip-python
python -c 'from hip import hip, hiprtc' 2> /dev/null && echo 'Success' || echo 'Failure'
```
HIP-Python has a great4dd capability for retrieving device properties and 
attributes. We'll take a look at the two main functions -- higGetDeviceProperties and
hipDeviceGetAttribute.

## Obtaining Device Properties

We'll take a look at the higGetDeviceProperties function first. Copy the following 
code into a file named `hipGetDevicePropeties_example.py` or pull the example down
with 

```
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/Python/hip-python
```

```
from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))

for attrib in sorted(props.PROPERTIES()):
    print(f"props.{attrib}={getattr(props,attrib)}")
print("ok")
```

Run it by loading the proper modules and running it with python3.

```
module load rocm hip-python
python3 hipGetDeviceProperties_example.py
```

Some of the useful properties that can be obtained are:

```
props.managedMemory=1
props.name=b'AMD Instinct MI210'
props.warpSize=64
```

## Getting Device Attributes

The second function to get device information is hipDeviceGetAttribute.
Copy the following into `hipDeviceGetAttribute_example.py` or use the
file in the hip-python examples.

```
from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

device_num = 0

for attrib in (
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeWarpSize,
):
    value = hip_check(hip.hipDeviceGetAttribute(attrib,device_num))
    print(f"{attrib.name}: {value}")
print("ok")
```

Run this file.

```
module load rocm hip-python
python3 hipDeviceGetAttribute_example.py
```

Output 

```
hipDeviceAttributeMaxBlockDimX: 1024
hipDeviceAttributeMaxBlockDimY: 1024
hipDeviceAttributeMaxBlockDimZ: 1024
hipDeviceAttributeMaxGridDimX: 2147483647
hipDeviceAttributeMaxGridDimY: 65536
hipDeviceAttributeMaxGridDimZ: 65536
hipDeviceAttributeWarpSize: 64
ok
```

# Accessing HIP Streams using HIP-Python



```
import ctypes
import random
import array

from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

# inputs
n = 100
x_h = array.array("i",[int(random.random()*10) for i in range(0,n)])
num_bytes = x_h.itemsize * len(x_h)
x_d = hip_check(hip.hipMalloc(num_bytes))

stream = hip_check(hip.hipStreamCreate())
hip_check(hip.hipMemcpyAsync(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice,stream))
hip_check(hip.hipMemsetAsync(x_d,0,num_bytes,stream))
hip_check(hip.hipMemcpyAsync(x_h,x_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost,stream))
hip_check(hip.hipStreamSynchronize(stream))
hip_check(hip.hipStreamDestroy(stream))

# deallocate device data 
hip_check(hip.hipFree(x_d))

for i,x in enumerate(x_h):
    if x != 0:
        raise ValueError(f"expected '0' for element {i}, is: '{x}'")
print("ok")
```

# Compiling and Launching Kernels

```
from hip import hip, hiprtc

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


source = b"""\
extern "C" __global__ void print_tid() {
  printf("tid: %d\\n", (int) threadIdx.x);
}
"""

prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"print_tid", 0, [], []))

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))
arch = props.gcnArchName

print(f"Compiling kernel for {arch}")

cflags = [b"--offload-arch="+arch]
err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
    log = bytearray(log_size)
    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
    raise RuntimeError(log.decode())
code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code))
module = hip_check(hip.hipModuleLoadData(code))
kernel = hip_check(hip.hipModuleGetFunction(module, b"print_tid"))
#
hip_check(
    hip.hipModuleLaunchKernel(
        kernel,
        *(1, 1, 1), # grid
        *(32, 1, 1),  # block
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=None,
    )
)

hip_check(hip.hipDeviceSynchronize())
hip_check(hip.hipModuleUnload(module))
hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

print("ok")
```

## Kernels with arguments

```
import ctypes
import array
import random
import math

from hip import hip, hiprtc

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

source = b"""\
extern "C" __global__ void scale_vector(float factor, int n, short unused1, int unused2, float unused3, float *x) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid == 0 ) {
    printf("tid: %d, factor: %f, x*: %lu, n: %lu, unused1: %d, unused2: %d, unused3: %f\\n",tid,factor,x,n,(int) unused1,unused2,unused3);
  }
  if (tid < n) {
     x[tid] *= factor;
  }
}
"""

prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"scale_vector", 0, [], []))

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))
arch = props.gcnArchName

print(f"Compiling kernel for {arch}")

cflags = [b"--offload-arch="+arch]
err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
    log = bytearray(log_size)
    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
    raise RuntimeError(log.decode())
code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code))
module = hip_check(hip.hipModuleLoadData(code))
kernel = hip_check(hip.hipModuleGetFunction(module, b"scale_vector"))

# kernel launch

## inputs
n = 100
x_h = array.array("f",[random.random() for i in range(0,n)])
num_bytes = x_h.itemsize * len(x_h)
x_d = hip_check(hip.hipMalloc(num_bytes))
print(f"{hex(int(x_d))=}")

## upload host data
hip_check(hip.hipMemcpy(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))

factor = 1.23

## expected result
x_expected = [a*factor for a in x_h]

block = hip.dim3(x=32)
grid = hip.dim3(math.ceil(n/block.x))

## launch
hip_check(
    hip.hipModuleLaunchKernel(
        kernel,
        *grid,
        *block,
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=( 
          ctypes.c_float(factor), # 4 bytes
          ctypes.c_int(n),  # 8 bytes
          ctypes.c_short(5), # unused1, 10 bytes
          ctypes.c_int(2), # unused2, 16 bytes (+2 padding bytes)
          ctypes.c_float(5.6), # unused3 20 bytes
          x_d, # 32 bytes (+4 padding bytes)
        )
    )
)

# copy result back
hip_check(hip.hipMemcpy(x_h,x_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost))

for i,x_h_i in enumerate(x_h):
    if not math.isclose(x_h_i,x_expected[i],rel_tol=1e-6):
        raise RuntimeError(f"values do not match, {x_h[i]=} vs. {x_expected[i]=}, {i=}")

hip_check(hip.hipFree(x_d))

hip_check(hip.hipModuleUnload(module))
hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

print("ok")
```

## Calling hipBLAS from Python using HIP-Python

In file hipblas_numpy_example.py
```
import ctypes
import math
import numpy as np

from hip import hip
from hip import hipblas

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err,hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err,hipblas.hipblasStatus_t) and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    return result

num_elements = 100

# input data on host
alpha = ctypes.c_float(2)
x_h = np.random.rand(num_elements).astype(dtype=np.float32)
y_h = np.random.rand(num_elements).astype(dtype=np.float32)

# expected result
y_expected = alpha*x_h + y_h

# device vectors
num_bytes = num_elements * np.dtype(np.float32).itemsize
x_d = hip_check(hip.hipMalloc(num_bytes))
y_d = hip_check(hip.hipMalloc(num_bytes))

# copy input data to device
hip_check(hip.hipMemcpy(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(y_d,y_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))

# call hipblasSaxpy + initialization & destruction of handle
handle = hip_check(hipblas.hipblasCreate())
hip_check(hipblas.hipblasSaxpy(handle, num_elements, ctypes.addressof(alpha), x_d, 1, y_d, 1))
hip_check(hipblas.hipblasDestroy(handle))

# copy result (stored in y_d) back to host (store in y_h)
hip_check(hip.hipMemcpy(y_h,y_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# compare to expected result
if np.allclose(y_expected,y_h):
    print("ok")
else:
    print("FAILED")
#print(f"{y_h=}")
#print(f"{y_expected=}")

# clean up
hip_check(hip.hipFree(x_d))
hip_check(hip.hipFree(y_d))
```

import numpy as np
from hip import hip, hipfft

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    if isinstance(err, hipfft.hipfftResult) and err != hipfft.hipfftResult.HIPFFT_SUCCESS:
        raise RuntimeError(str(err))
    return result

# initial data
N = 100
hx = np.zeros(N,dtype=np.cdouble)
hx[:] = 1 - 1j

# copy to device
dx = hip_check(hip.hipMalloc(hx.size*hx.itemsize))
hip_check(hip.hipMemcpy(dx, hx, dx.size, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# create plan
plan = hip_check(hipfft.hipfftPlan1d(N, hipfft.hipfftType.HIPFFT_Z2Z, 1))

# execute plan
hip_check(hipfft.hipfftExecZ2Z(plan, idata=dx, odata=dx, direction=hipfft.HIPFFT_FORWARD))
hip_check(hip.hipDeviceSynchronize())

# copy to host and free device data
hip_check(hip.hipMemcpy(hx,dx,dx.size,hip.hipMemcpyKind.hipMemcpyDeviceToHost))
hip_check(hip.hipFree(dx))

if not np.isclose(hx[0].real,N) or not np.isclose(hx[0].imag,-N):
     raise RuntimeError("element 0 must be '{N}-j{N}'.")
for i in range(1,N):
   if not np.isclose(abs(hx[i]),0):
        raise RuntimeError(f"element {i} must be '0'")

hip_check(hipfft.hipfftDestroy(plan))
print("ok")
```

## Calling hipFFT from Python using HIP-Python

```
import numpy as np
from hip import hip, hipfft

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    if isinstance(err, hipfft.hipfftResult) and err != hipfft.hipfftResult.HIPFFT_SUCCESS:
        raise RuntimeError(str(err))
    return result

# initial data
N = 100
hx = np.zeros(N,dtype=np.cdouble)
hx[:] = 1 - 1j

# copy to device
dx = hip_check(hip.hipMalloc(hx.size*hx.itemsize))
hip_check(hip.hipMemcpy(dx, hx, dx.size, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# create plan
plan = hip_check(hipfft.hipfftPlan1d(N, hipfft.hipfftType.HIPFFT_Z2Z, 1))

# execute plan
hip_check(hipfft.hipfftExecZ2Z(plan, idata=dx, odata=dx, direction=hipfft.HIPFFT_FORWARD))
hip_check(hip.hipDeviceSynchronize())

# copy to host and free device data
hip_check(hip.hipMemcpy(hx,dx,dx.size,hip.hipMemcpyKind.hipMemcpyDeviceToHost))
hip_check(hip.hipFree(dx))

if not np.isclose(hx[0].real,N) or not np.isclose(hx[0].imag,-N):
     raise RuntimeError("element 0 must be '{N}-j{N}'.")
for i in range(1,N):
   if not np.isclose(abs(hx[i]),0):
        raise RuntimeError(f"element {i} must be '0'")

hip_check(hipfft.hipfftDestroy(plan))
print("ok")
```


## Calling RCCL from Python using HIP-Python

```
import numpy as np
from hip import hip, rccl

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    if isinstance(err, rccl.ncclResult_t) and err != rccl.ncclResult_t.ncclSuccess:
        raise RuntimeError(str(err))
    return result

# init the communicators
num_gpus = hip_check(hip.hipGetDeviceCount())
comms = np.empty(num_gpus,dtype="uint64") # size of pointer type, such as ncclComm
devlist = np.array(range(0,num_gpus),dtype="int32")
hip_check(rccl.ncclCommInitAll(comms, num_gpus, devlist))

# init data on the devices
N = 4
ones = np.ones(N,dtype="int32")
zeros = np.zeros(ones.size,dtype="int32")
dxlist = []
for dev in devlist:
    hip_check(hip.hipSetDevice(dev))
    dx = hip_check(hip.hipMalloc(ones.size*ones.itemsize)) # items are bytes
    dxlist.append(dx)
    hx = ones if dev == 0 else zeros
    hip_check(hip.hipMemcpy(dx,hx,dx.size,hip.hipMemcpyKind.hipMemcpyHostToDevice))

# perform a broadcast
hip_check(rccl.ncclGroupStart())
for dev in devlist:
    hip_check(hip.hipSetDevice(dev))
    hip_check(rccl.ncclBcast(dxlist[dev], N, rccl.ncclDataType_t.ncclInt32, 0, int(comms[dev]), None)) 
    # conversion to Python int is required to not let the numpy datatype to be interpreted as single-element Py_buffer
hip_check(rccl.ncclGroupEnd())

# download and check the output; confirm all entries are one
hx = np.empty(N,dtype="int32")
for dev in devlist:
    dx=dxlist[dev]
    hx[:] = 0
    hip_check(hip.hipMemcpy(hx,dx,dx.size,hip.hipMemcpyKind.hipMemcpyDeviceToHost)) 
    for i,item in enumerate(hx):
        if item != 1:
            raise RuntimeError(f"failed for element {i}")

# clean up
for dx in dxlist:
    hip_check(hip.hipFree(dx))
for comm in comms:
    hip_check(rccl.ncclCommDestroy(int(comm)))
    # conversion to Python int is required to not let the numpy datatype to be interpreted as single-element Py_buffer

print("ok")
```
