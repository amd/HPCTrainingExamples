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
