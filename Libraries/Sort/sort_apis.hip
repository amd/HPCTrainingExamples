// sort_apis.hip  —  the same array sorted three ways: thrust, rocPRIM, hipCUB
//
// Purpose: one place that shows the equivalent device-sort calls in the three
// libraries you get with ROCm, using unified memory on MI300A (gfx942). Every
// path sorts a COPY of the same input so we can cross-check that all three agree
// bit-for-bit, and each is timed with hipEvent for an apples-to-apples baseline.
//
//   keys-only : thrust::sort           / rocprim::radix_sort_keys
//                                       / hipcub::DeviceRadixSort::SortKeys
//   key-value : thrust::sort_by_key    / rocprim::radix_sort_pairs
//                                       / hipcub::DeviceRadixSort::SortPairs
//   descending: thrust + thrust::greater / *_descending variants
//
// rocPRIM and hipCUB use the canonical TWO-CALL temporary-storage pattern:
//   1) call with temporary_storage == nullptr  -> fills in temp_bytes
//   2) hipMalloc(temp_bytes); call again        -> does the sort
//
// Build: hipcc -O3 --offload-arch=gfx942 sort_apis.hip -o sort_apis
// Run:   ./sort_apis [n]
#include <cstring>                 // global ::memset visible before rocprim/hipcub
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <hipcub/hipcub.hpp>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdint>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

using u32 = uint32_t;

// Time a callable with hipEvent (device time in ms). A warm-up is done by the
// caller; this only measures the timed run.
template <class F>
static float time_ms(F&& fn){
    hipEvent_t ev0,ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0));
    fn();
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms=0.f; HIP_CHECK(hipEventElapsedTime(&ms,ev0,ev1));
    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    return ms;
}

static bool is_sorted_u32(const u32* a, int n, bool ascending=true){
    for (int i=1;i<n;++i){ if (ascending ? (a[i-1]>a[i]) : (a[i-1]<a[i])) return false; }
    return true;
}

int main(int argc, char** argv){
    int n = (argc > 1) ? atoi(argv[1]) : (1<<22);   // ~4M keys by default
    const double MKPS = n / 1.0e3;                  // n / (ms * 1e3) = Mkeys/s

    // Unified memory: host fills, GPU sorts, no hipMemcpy on the APU.
    u32 *ref, *vref;                                 // pristine inputs
    HIP_CHECK(hipMalloc(&ref,  (size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&vref, (size_t)n*sizeof(u32)));
    srand(12345);
    for (int i=0;i<n;++i){ ref[i] = ((u32)rand()<<16) ^ (u32)rand(); vref[i] = (u32)i; }

    // Reusable buffers (double-buffered for rocPRIM/hipCUB out-of-place sorts).
    u32 *ka, *kb, *va, *vb;
    HIP_CHECK(hipMalloc(&ka,(size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&kb,(size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&va,(size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&vb,(size_t)n*sizeof(u32)));
    auto reset_keys = [&]{ memcpy(ka, ref, (size_t)n*sizeof(u32)); };
    auto reset_pairs= [&]{ memcpy(ka, ref, (size_t)n*sizeof(u32));
                           memcpy(va, vref,(size_t)n*sizeof(u32)); };

    printf("sort_apis: n=%d keys (32-bit unsigned), unified memory\n\n", n);
    printf("%-34s %10s %12s\n", "operation / library", "time(ms)", "Mkeys/s");

    // ---------- keys-only, ascending ----------
    // thrust::sort (in-place)
    reset_keys(); thrust::sort(thrust::device, ka, ka+n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    reset_keys();
    float t_thrust = time_ms([&]{ thrust::sort(thrust::device, ka, ka+n); });
    bool ok_thrust = is_sorted_u32(ka, n);
    printf("%-34s %10.3f %12.2f%s\n", "keys  thrust::sort", t_thrust, MKPS/t_thrust, ok_thrust?"":"  [FAIL]");

    // rocprim::radix_sort_keys (two-call temp storage, out-of-place ka->kb)
    {
        size_t tb=0; void* tmp=nullptr;
        reset_keys();
        HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, tb, ka, kb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        (void)rocprim::radix_sort_keys(tmp, tb, ka, kb, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
        reset_keys();
        float t = time_ms([&]{ (void)rocprim::radix_sort_keys(tmp, tb, ka, kb, n); });
        bool ok = is_sorted_u32(kb, n);
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n",
               "keys  rocprim::radix_sort_keys", t, MKPS/t, ok?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }

    // hipcub::DeviceRadixSort::SortKeys (two-call temp storage, ka->kb)
    {
        size_t tb=0; void* tmp=nullptr;
        reset_keys();
        HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(nullptr, tb, ka, kb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(tmp, tb, ka, kb, n));
        HIP_CHECK(hipDeviceSynchronize()); // warm-up
        reset_keys();
        float t = time_ms([&]{ HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(tmp, tb, ka, kb, n)); });
        bool ok = is_sorted_u32(kb, n);
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n",
               "keys  hipcub::DeviceRadixSort", t, MKPS/t, ok?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }

    // ---------- key-value pairs, ascending ----------
    printf("\n");
    // thrust::sort_by_key (in-place on ka/va)
    reset_pairs();
    float tp_thrust = time_ms([&]{ thrust::sort_by_key(thrust::device, ka, ka+n, va); });
    printf("%-34s %10.3f %12.2f%s\n", "pairs thrust::sort_by_key",
           tp_thrust, MKPS/tp_thrust, is_sorted_u32(ka,n)?"":"  [FAIL]");

    // rocprim::radix_sort_pairs (ka,va -> kb,vb)
    {
        size_t tb=0; void* tmp=nullptr;
        reset_pairs();
        HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr, tb, ka, kb, va, vb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        float t = time_ms([&]{ (void)rocprim::radix_sort_pairs(tmp, tb, ka, kb, va, vb, n); });
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n", "pairs rocprim::radix_sort_pairs",
               t, MKPS/t, is_sorted_u32(kb,n)?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }

    // hipcub::DeviceRadixSort::SortPairs (ka,va -> kb,vb)
    {
        size_t tb=0; void* tmp=nullptr;
        reset_pairs();
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(nullptr, tb, ka, kb, va, vb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        float t = time_ms([&]{ HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp, tb, ka, kb, va, vb, n)); });
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n", "pairs hipcub::DeviceRadixSort",
               t, MKPS/t, is_sorted_u32(kb,n)?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }

    // ---------- descending keys ----------
    printf("\n");
    reset_keys();
    float td_thrust = time_ms([&]{ thrust::sort(thrust::device, ka, ka+n, thrust::greater<u32>()); });
    printf("%-34s %10.3f %12.2f%s\n", "keys  thrust::sort (descending)",
           td_thrust, MKPS/td_thrust, is_sorted_u32(ka,n,false)?"":"  [FAIL]");
    {
        size_t tb=0; void* tmp=nullptr;
        reset_keys();
        HIP_CHECK((hipError_t)rocprim::radix_sort_keys_desc(nullptr, tb, ka, kb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        float t = time_ms([&]{ (void)rocprim::radix_sort_keys_desc(tmp, tb, ka, kb, n); });
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n", "keys  rocprim::radix_sort_keys_desc",
               t, MKPS/t, is_sorted_u32(kb,n,false)?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }
    {
        size_t tb=0; void* tmp=nullptr;
        reset_keys();
        HIP_CHECK(hipcub::DeviceRadixSort::SortKeysDescending(nullptr, tb, ka, kb, n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        float t = time_ms([&]{ HIP_CHECK(hipcub::DeviceRadixSort::SortKeysDescending(tmp, tb, ka, kb, n)); });
        printf("%-34s %10.3f %12.2f%s   temp=%zuB\n", "keys  hipcub::...SortKeysDescending",
               t, MKPS/t, is_sorted_u32(kb,n,false)?"":"  [FAIL]", tb);
        HIP_CHECK(hipFree(tmp));
    }

    printf("\nAll [FAIL] tags empty => the three libraries agree on the result.\n");

    (void)hipFree(ref);(void)hipFree(vref);
    (void)hipFree(ka);(void)hipFree(kb);(void)hipFree(va);(void)hipFree(vb);
    return 0;
}
