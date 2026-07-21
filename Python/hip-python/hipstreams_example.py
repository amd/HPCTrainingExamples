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
