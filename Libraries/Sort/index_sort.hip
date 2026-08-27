// Copyright AMD 2026, MIT License
// Author: Bob Robey Bob.Robey@amd.com with AI tool help
//
// index_sort.hip  —  index sort (argsort) vs. sorting the data itself
//
// Often you do NOT want to move the records; you want a PERMUTATION that puts
// them in order. That is an "index sort" (a.k.a. argsort): sort an index array
// [0,1,2,...] using the keys, leaving the payload untouched. You then either
// (a) keep the permutation and read records indirectly, or (b) GATHER once to
// materialize a sorted copy.
//
// This contrasts two strategies on the SAME keys and a wide payload:
//   1) INDEX sort : sort (key,index) pairs, payload stays put; optional gather.
//   2) DATA  sort : sort (key,payload-word) pairs, moving the payload directly.
//
// Both use library radix sorts (rocPRIM / hipCUB / thrust). The teaching point:
// when the payload is large, sorting a 4-byte index + one gather beats dragging
// the whole record through every radix pass; when the payload is a single word,
// sorting the data directly is simplest. name_sort.hip already does an implicit
// index sort (thrust::sort_by_key on id[]); this makes it explicit and measured.
//
// Build: hipcc -O3 --offload-arch=gfx942 index_sort.hip -o index_sort
// Run:   ./index_sort [n] [payload_words]
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <hipcub/hipcub.hpp>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdint>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

using u32 = uint32_t;

template <class F>
static float time_ms(F&& fn){
    hipEvent_t ev0,ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0)); fn(); HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms=0.f; HIP_CHECK(hipEventElapsedTime(&ms,ev0,ev1));
    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    return ms;
}

// Gather: out[i] = payload[ perm[i] ]  (materialize records in sorted order).
// PW = payload words per record.
__global__ void gather_records(const u32* __restrict__ perm, int n, int pw,
                               const u32* __restrict__ payload, u32* __restrict__ out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        const u32* src = payload + (size_t)perm[i]*pw;
        u32*       dst = out     + (size_t)i*pw;
        for (int w=0; w<pw; ++w) dst[w] = src[w];
    }
}

int main(int argc, char** argv){
    int n  = (argc > 1) ? atoi(argv[1]) : (1<<22);
    int pw = (argc > 2) ? atoi(argv[2]) : 8;      // payload words per record
    if (pw < 1) pw = 1;
    const double MKPS = n / 1.0e3;

    // Unified memory. keys + a wide payload (pw 32-bit words/record).
    u32 *keys, *payload;
    HIP_CHECK(hipMalloc(&keys,    (size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&payload, (size_t)n*pw*sizeof(u32)));
    srand(2026);
    for (int i=0;i<n;++i){
        keys[i] = ((u32)rand()<<16) ^ (u32)rand();
        for (int w=0; w<pw; ++w) payload[(size_t)i*pw+w] = (u32)i;  // record id in every word
    }

    // Scratch for the two strategies.
    u32 *kin,*kout,*idx_in,*idx_out,*sorted_payload;
    HIP_CHECK(hipMalloc(&kin,   (size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&kout,  (size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&idx_in,(size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&idx_out,(size_t)n*sizeof(u32)));
    HIP_CHECK(hipMalloc(&sorted_payload,(size_t)n*pw*sizeof(u32)));

    printf("index_sort: n=%d  payload=%d words (%zu B/record)\n\n",
           n, pw, (size_t)pw*sizeof(u32));
    printf("%-40s %10s %12s\n", "strategy", "time(ms)", "Mkeys/s");

    int T=256, B=(n+T-1)/T;

    // ---------- Strategy 1: INDEX sort (argsort) ----------
    // Sort (key, index) pairs; payload never moves. Then gather once.
    auto init_idx = [&]{ for (int i=0;i<n;++i){ idx_in[i]=(u32)i; } memcpy(kin,keys,(size_t)n*sizeof(u32)); };

    // One warm-up sort so the first timed library call below is not charged for
    // GPU/module initialization (first hipMalloc, first kernel launch, etc.).
    {
        size_t tb=0; void* tmp=nullptr; init_idx();
        HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr,tb,kin,kout,idx_in,idx_out,n));
        HIP_CHECK(hipMalloc(&tmp, tb?tb:1));
        (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,idx_in,idx_out,n);
        gather_records<<<B,T>>>(idx_out,n,pw,payload,sorted_payload);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(tmp));
    }

    // rocPRIM argsort
    {
        size_t tb=0; void* tmp=nullptr;
        init_idx();
        HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr,tb,kin,kout,idx_in,idx_out,n));
        HIP_CHECK(hipMalloc(&tmp, tb?tb:1));
        float t_sort = time_ms([&]{ (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,idx_in,idx_out,n); });
        float t_gath = time_ms([&]{ gather_records<<<B,T>>>(idx_out,n,pw,payload,sorted_payload); });
        HIP_CHECK(hipDeviceSynchronize());
        // verify: sorted keys ascending AND gathered record id matches permutation
        bool ok=true; for(int i=1;i<n&&ok;++i) if(kout[i-1]>kout[i]) ok=false;
        for(int i=0;i<n&&ok;++i) if(sorted_payload[(size_t)i*pw]!=idx_out[i]) ok=false;
        printf("%-40s %10.3f %12.2f%s\n", "1a idx rocprim pairs (sort only)",
               t_sort, MKPS/t_sort, ok?"":"  [FAIL]");
        printf("%-40s %10.3f %12.2f\n",   "   + gather payload = total",
               t_sort+t_gath, MKPS/(t_sort+t_gath));
        HIP_CHECK(hipFree(tmp));
    }
    // hipCUB argsort
    {
        size_t tb=0; void* tmp=nullptr;
        init_idx();
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(nullptr,tb,kin,kout,idx_in,idx_out,n));
        HIP_CHECK(hipMalloc(&tmp, tb?tb:1));
        float t = time_ms([&]{ HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp,tb,kin,kout,idx_in,idx_out,n)); });
        bool ok=true; for(int i=1;i<n&&ok;++i) if(kout[i-1]>kout[i]) ok=false;
        printf("%-40s %10.3f %12.2f%s\n", "1b idx hipcub pairs (sort only)",
               t, MKPS/t, ok?"":"  [FAIL]");
        HIP_CHECK(hipFree(tmp));
    }
    // thrust argsort (in-place on kin + idx_in)
    {
        init_idx();
        float t = time_ms([&]{ thrust::sort_by_key(thrust::device, kin, kin+n, idx_in); });
        bool ok=true; for(int i=1;i<n&&ok;++i) if(kin[i-1]>kin[i]) ok=false;
        printf("%-40s %10.3f %12.2f%s\n", "1c idx thrust::sort_by_key (sort only)",
               t, MKPS/t, ok?"":"  [FAIL]");
    }

    // ---------- Strategy 2: DATA sort (single payload word) ----------
    // Sort (key, first-payload-word) directly: no gather, but moves the payload
    // through the sort. Cheap here (1 word); expensive as pw grows.
    printf("\n");
    {
        size_t tb=0; void* tmp=nullptr;
        memcpy(kin,keys,(size_t)n*sizeof(u32));
        for (int i=0;i<n;++i) idx_in[i]=payload[(size_t)i*pw]; // move the record's word
        HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr,tb,kin,kout,idx_in,idx_out,n));
        HIP_CHECK(hipMalloc(&tmp, tb?tb:1));
        float t = time_ms([&]{ (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,idx_in,idx_out,n); });
        bool ok=true; for(int i=1;i<n&&ok;++i) if(kout[i-1]>kout[i]) ok=false;
        printf("%-40s %10.3f %12.2f%s\n", "2  data rocprim pairs (1 word payload)",
               t, MKPS/t, ok?"":"  [FAIL]");
        HIP_CHECK(hipFree(tmp));
    }

    printf("\nWith a %d-word payload, index-sort + gather moves 4 B/key through the\n"
           "radix passes and copies the record once; a full data sort would move\n"
           "%zu B/key through every pass -- the gap widens with payload size.\n",
           pw, (size_t)pw*sizeof(u32));

    (void)hipFree(keys);(void)hipFree(payload);
    (void)hipFree(kin);(void)hipFree(kout);(void)hipFree(idx_in);(void)hipFree(idx_out);
    (void)hipFree(sorted_payload);
    return 0;
}
