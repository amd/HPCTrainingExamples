// rocprim_tempsize.hip
// Demonstrates the memory story behind the "circular buffer" radix work:
// rocPRIM's Onesweep device_radix_sort needs a temporary buffer whose size
// grows with the number of keys n (the decoupled-look-back state). We do NOT
// sort here -- we only ask rocPRIM how much temporary storage it *would* need,
// by calling radix_sort_keys with a null temporary_storage pointer.
//
// Contrast (from the GPUOpen blog / GPU Zen 3): Orochi's circular-buffer
// extension holds that look-back buffer at a constant ~2 MB regardless of n.
//
// Build:  hipcc -O3 --offload-arch=gfx942 rocprim_tempsize.hip -o rocprim_tempsize
// Run:    ./rocprim_tempsize

#include <cstring>   // ensure global ::memset is visible before rocprim headers
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdint>

int main(){
    const size_t sizes[] = { 1'000'000ull, 10'000'000ull, 100'000'000ull,
                             500'000'000ull, 1'000'000'000ull };

    // Dummy device pointers are fine: with temporary_storage == nullptr,
    // rocPRIM only computes the required storage size and does not dereference.
    unsigned int* d_in  = nullptr;
    unsigned int* d_out = nullptr;

    printf("%14s  %16s  %10s\n", "n (keys)", "temp_storage (B)", "bytes/key");
    for (size_t n : sizes){
        size_t temp_bytes = 0;
        hipError_t st = (hipError_t)rocprim::radix_sort_keys(
            nullptr, temp_bytes, d_in, d_out, n);
        if (st != hipSuccess){
            printf("n=%zu: query failed (%s)\n", n, hipGetErrorString(st));
            continue;
        }
        printf("%14zu  %16zu  %10.3f\n",
               n, temp_bytes, (double)temp_bytes / (double)n);
    }
    return 0;
}
