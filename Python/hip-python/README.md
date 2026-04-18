
# HIP-Python

README.md from `HPCTrainingExamples/Python/hip-python` in the Training Examples repository

For these examples, get a GPU with salloc or srun.

```bash
salloc -N 1 --ntasks 16 --gpus=1 --time=01:00:00
or
srun -N 1 --ntasks 16 --gpus=1 --time=01:00:00 --pty /bin/bash
```

Be sure and free up the GPU when you are done with the exercises.

The first test is to check that the hip-python environment is set up correctly.

```bash
module load rocm hip-python
python -c 'from hip import hip, hiprtc' 2> /dev/null && echo 'Success' || echo 'Failure'
```

> [!NOTE]
> These examples assume you are working on AAC6, where a `hip-python` module is already
> installed. On your own or other training systems such as AAC7, you can install it yourself with
> ```bash
> python3 -m venv hip-python-venv
> source hip-python-venv/bin/activate
> python3 -m pip install -i https://test.pypi.org/simple hip-python~=7.2.0
> python3 -m pip install -i https://test.pypi.org/simple hip-python-as-cuda~=7.2.0
> ```
> Make sure to specify the **correct version** string that matches your **local ROCm installation**.
> This example installs `hip-python` for `rocm/7.2.0`.

HIP-Python has an extensive capability for retrieving device properties and 
attributes. We'll take a look at the two main functions -- higGetDeviceProperties and
hipDeviceGetAttribute.

## Obtaining Device Properties

We'll take a look at the higGetDeviceProperties function first. Copy the following 
code into a file named `hipGetDeviceProperties_example.py` or pull the example down
with 

```bash
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/Python/hip-python
```

The `hipGetDeviceProperties_example.py` file

```python
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

Try it by loading the proper modules and running it with python3.

```bash
module load rocm hip-python
python3 hipGetDeviceProperties_example.py
```

Some of the useful properties that can be obtained are:

```text
props.managedMemory=1
props.name=b'AMD Instinct MI210'
props.warpSize=64
```

## Getting Device Attributes

The second function to get device information is hipDeviceGetAttribute.
Copy the following into `hipDeviceGetAttribute_example.py` or use the
file in the hip-python examples.

```python
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

```bash
module load rocm hip-python
python3 hipDeviceGetAttribute_example.py
```

Output 

```text
hipDeviceAttributeMaxBlockDimX: 1024
hipDeviceAttributeMaxBlockDimY: 1024
hipDeviceAttributeMaxBlockDimZ: 1024
hipDeviceAttributeMaxGridDimX: 2147483647
hipDeviceAttributeMaxGridDimY: 65536
hipDeviceAttributeMaxGridDimZ: 65536
hipDeviceAttributeWarpSize: 64
ok
```

## Accessing HIP Streams using HIP-Python

In the HIP streams example, we'll see how to create streams from Python
and pass array data to the stream routines from Python arrays.

The code in the file hipstreams_example.py.

```python
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

Now run this example.

```bash
module load rocm hip-python
python3 hipstreams_example.py
```

## Calling hipBLAS from Python using HIP-Python

In the file hipblas_numpy_example.py, the hipBLAS library
Saxpy routine is called. It operates on a numpy data
array.

```python
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

## Using Unified Shared Memory for hipBLAS using HIP-Python

We can also take advantage of the single address space on the MI300A or the
managed memory that moves the data from host to device and back for us on 
the other AMD Instinct GPUs. It simplifies the code because the memory
does not have to be duplicated on the CPU and GPU. The code is in the
file hipblas_numpy_USM_example.py.

```python
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

# call hipblasSaxpy + initialization & destruction of handle
handle = hip_check(hipblas.hipblasCreate())
hip_check(hipblas.hipblasSaxpy(handle, num_elements, ctypes.addressof(alpha), x_h, 1, y_h, 1))
hip_check(hipblas.hipblasDestroy(handle))

# compare to expected result
if np.allclose(y_expected,y_h):
    print("ok")
else:
    print("FAILED")
#print(f"{y_h=}")
#print(f"{y_expected=}")
```

To run this unified shared memory example, we also
need the environment variable `HSA_XNACK` set to one.

```bash
module load rocm hip-python
export HSA_XNACK=1
python3 hipblas_numpy_USM_example.py
```

## Calling hipFFT from Python using HIP-Python

The HIP FFT library can also be called from Python. We create a plan, perform
the FFT, and then destroy the plan. This file is `hipfft_numpy_example.py`.

```python
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

Run this examples with:

```bash
module load rocm hip-python
python3 hipfft_numpy_example.py
```

## Unified Shared Memory version of calling hipFFT HIP-Python

The code is much simplier if we take advantage of the unified shared memory
or managed memory. We can just use the host versions of the data directly.
The simpler code is in hipfft_numpy_USM_example.py

```python
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

# create plan
plan = hip_check(hipfft.hipfftPlan1d(N, hipfft.hipfftType.HIPFFT_Z2Z, 1))

# execute plan
hip_check(hipfft.hipfftExecZ2Z(plan, idata=hx, odata=hx, direction=hipfft.HIPFFT_FORWARD))
hip_check(hip.hipDeviceSynchronize())

if not np.isclose(hx[0].real,N) or not np.isclose(hx[0].imag,-N):
     raise RuntimeError("element 0 must be '{N}-j{N}'.")
for i in range(1,N):
   if not np.isclose(abs(hx[i]),0):
        raise RuntimeError(f"element {i} must be '0'")

hip_check(hipfft.hipfftDestroy(plan))
print("ok")
```

Run this with:

```bash
module load rocm hip-python
export HSA_XNACK=1
python3 hipfft_numpy_USM_example.py
```

## Calling RCCL from Python using HIP-Python

We can also call the RCCL communication library from Python using
HIP-Python. An example of this is shown in rccl_example.py.

```python
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

Running this example:

```bash
module load rocm hip-python
python3 rccl_example.py
```

## Unified Shared Memory with RCCL using HIP-Python

We can also use the host data directly by relying on the unified shared memory
or the managed memory on the AMD Instinct GPUs. The code for this is shown
in rccl_USM_example.py

```python
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
    hx = ones if dev == 0 else zeros
    dxlist.append(hx)

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
    hx=dxlist[dev]
    for i,item in enumerate(hx):
        if item != 1:
            raise RuntimeError(f"failed for element {i}")

# clean up
for comm in comms:
    hip_check(rccl.ncclCommDestroy(int(comm)))
    # conversion to Python int is required to not let the numpy datatype to be interpreted as single-element Py_buffer

print("ok")
```

Running this version requires setting `HSA_XNACK` to one as in the previous unified shared memory examples.

```bash
module load rocm hip-python
export HSA_XNACK=1
python3 rccl_USM_example.py
```

## Cython Basics

Cython compiles Python-like code to C for significant performance gains on CPU-bound operations.
This section demonstrates basic Cython usage with a simple array sum example.

The example is in the `cython_basic/` directory.

### Simple Array Sum Example

The file `array_sum.pyx` contains the function that we like
to pre-precompile. It is written in Cython syntax, which adds a few
additional features to standard Python such as `cdef` to define types.

```cython
def array_sum(double[:, ::1] A):
    """Compute the sum of all elements in a 2D array."""
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int i, j
    cdef double result = 0

    for i in range(m):
        for j in range(n):
            result += A[i, j]

    return result
```

The `setup.py` file defines how to build the extension:

```python
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension("array_sum", ["array_sum.pyx"])],
        compiler_directives={"language_level": 3},
    )
)
```
The key routine `setup()` is the entry point for the compilation.
The `cythonize` utility takes care of translating the `*.pyx` file to
C code, while the other commands provide details on how the translated
files should be compiled, i.e. compiler flags, linked libraries,
the list of source files to compile (in this case only `array_sum.pyx`),
and the name of compiled module (in this case `array_sum`).

Once the extension has been compiled, it can be used in other
Python files like a regular Python module:
```python
import array_sum                 # Importing our compiled extension
result = array_sum.array_sum(A)  # Calling the precompiled function
```

### Build and Run

First, install `cython` in your environment with
```bash
python3 -m venv cython_venv
source cython_venv/bin/activate
python3 -m pip install cython numpy
```

Then, build the extension and run the demo
```bash
python3 setup.py build_ext --inplace
python3 cython_basic.py
```

Finally, clean up with
```bash
deactivate
rm -rf cython_venv build array_sum*.so array_sum.c
```

The output should look like something like this (speedup varies by system):

```text
Matrix size: 1000x1000
Pure Python: 113.0 ms (result: 500591.090701)
Cython:      0.8 ms (result: 500591.090701)
Speedup:     138.4x
Correctness verified!
ok
```

You can see the significant speedup we achieved by pre-compiling the
`array_sum` function with Cython.

## Cython with HIP-Python

This example demonstrates how Cython can be used together with HIP to accelerate
the data preparation on the host before we launch a kernel. This pattern can
be applied if the host side orchestration and preparation work becomes the
bottleneck in Python code with GPU acceleration.

The example is in the `cython_hip_example/` directory.

### Source Files

First, we again prepare a Cython module with the code we want to speedup in `matrix_prep.pyx`
In this case, it includes scaling a matrix by a scalar and copying the data to the GPU and back
by calling the HIP API via `hip-python`. The code adds decorators which allow Cython to produce
more optimized code.

```cython
# cython: language_level=3
cimport cython
import numpy as np
from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def scale_only_cython(double[:, ::1] A, double scale):
    """Cython-optimized matrix scaling (CPU only)."""
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int i, j

    for i in range(m):
        for j in range(n):
            A[i, j] *= scale

@cython.boundscheck(False)
@cython.wraparound(False)
def prepare_and_transfer(double[:, ::1] A, double scale):
    """
    Cython-optimized: scale matrix on CPU, then transfer to GPU.
    The nested loop is what Cython accelerates vs pure Python.
    """
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int i, j

    # CPU work: scale the matrix (Cython makes this fast)
    for i in range(m):
        for j in range(n):
            A[i, j] *= scale

    # GPU work: transfer to device
    num_bytes = A.shape[0] * A.shape[1] * sizeof(double)
    d_ptr = hip_check(hip.hipMalloc(num_bytes))
    hip_check(hip.hipMemcpy(d_ptr, A, num_bytes, 
                            hip.hipMemcpyKind.hipMemcpyHostToDevice))

    return d_ptr, num_bytes

def transfer_back_and_free(d_ptr, double[:, ::1] A, size_t num_bytes):
    """Copy results from GPU and free device memory."""
    hip_check(hip.hipMemcpy(A, d_ptr, num_bytes,
                            hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipFree(d_ptr))
```

Next, we need a `setup.py` script to compile the Cython module.
In this case, the script gets more complex, since it has to
be compiled and linked against the base ROCm installation similar
to what we would expect for "normal" C-code.

```python
import os
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

setup(
    ext_modules=cythonize([
        Extension(
            "matrix_prep",
            sources=["matrix_prep.pyx"],
            include_dirs=[np.get_include()],
            library_dirs=[os.path.join(ROCM_PATH, "lib")],
            libraries=["amdhip64"],
            extra_compile_args=["-D__HIP_PLATFORM_AMD__"],
        )
    ], compiler_directives={"language_level": 3})
)
```

### Build and Run (requires GPU)

First, make sure that your environment is setup
correctly. Now, we also need a ROCm installation:
```bash
python3 -m venv venv_cython
source venv_cython/bin/activate
module load rocm hip-python
python3 -m pip install cython numpy
```
Then, we can build the Cython module and run it:

```bash
python3 setup.py build_ext --inplace
python3 demo.py
```

You will see some output like this (performance will
vary depending on your system):

```text
1. CPU Computation Only (1000x1000 matrix scaling):
   Pure Python: 156.2 ms
   Cython:      0.7 ms
   Speedup:     239.0x

2. Full Pipeline (Cython prep + HIP transfer):
   Cython + HIP transfer: 210.1 ms

3. Correctness Check:
   Results match - verified!

ok
```
Again, we see significant speedup if we precompile the host code!

Don't forget to clean up afterwards:
```bash
deactivate
rm -rf venv_cython build matrix_prep*.so matrix_prep.c
```

## Compiling and Launching Kernels

We can also create our own C programs and compile them with the
hiprtc module for a Just-In-Time (JIT) compile capability. This
example shows a C routine called `print_tid()` that is encoded
as a string. The string is then converted into program source and
compiled. We use the ability to query the device parameters to
get the GPU architecture to compile for.


```python
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

To run the example of creating a kernel and launching it:

```bash
module load rocm hip-python
python3 create_launch_C_kernel.py
```

## Kernels with arguments

It is a little more complicated to launch a kernel with arguments. The program
is `scale_vector()` and it has six arguments. We add an "extra" field with the
six arguments as part of the launch kernel call. This example is in `kernel_with_arguments.py`.

```python
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

Run this example with:

```bash
module load rocm hip-python
python3 kernel_with_arguments.py
```


## Numba-HIP

Numba-HIP allows to write GPU kernels for AMD GPUs and Just-in-Time (JIT) compilation using native Python code.

### Installation

Numba-HIP is already installed on AAC6 as part of the `hip-python` module and can be loaded with
```
module load rocm hip-python
```
On other systems such as your own, you can install HIP-Python and Numba-HIP with
```bash
python3 -m venv hip-python-build
source hip-python-build/bin/activate
python3 -m pip install -i https://test.pypi.org/simple hip-python~=6.4.1
python3 -m pip config set global.extra-index-url https://test.pypi.org/simple
python3 -m pip install "numba-hip[rocm-6-4-1] @ git+https://github.com/ROCm/numba-hip.git"
```
Replace the `module load` commands in the following by sourcing this virtual environment.

> [!NOTE]
> Make sure to install the correct version that matches the ROCm installed on your system.
> For this, replace `6.4.1` and `rocm-6-4-1` accordingly (e.g., `7.0.0` and `rocm-7-0-0`).


### Kernel Definition

The kernel uses the `@hip.jit` decorator and follows the standard GPU programming model:

```python
@hip.jit
def f(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = hip.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]
```

### Memory Management with `to_device`

Numba-HIP provides the `to_device` API to transfer NumPy arrays (such as `{a,b,c}_host`) to GPU memory:

```python
# Transfer to GPU memory via Numba-HIP API
a_dev = hip.to_device(a_host)
b_dev = hip.to_device(b_host)
c_dev = hip.to_device(c_host)
```

After kernel execution, copy results back with `copy_to_host()`:

```python
c_host = c_dev.copy_to_host()
```

### Running the Example

```bash
module load rocm hip-python
python3 numba-hip.py
```

### CUDA-Posing Mode

An alternative approach for porting existing CUDA code is to have numba-hip pose as CUDA.
This allows using `@cuda.jit` syntax on AMD GPUs:

```python
from numba import hip
hip.pose_as_cuda()
from numba import cuda

@cuda.jit
def f(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = cuda.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]
```

Running this example:

```bash
module load rocm hip-python
python3 numba-hip-cuda-posing.py
```
