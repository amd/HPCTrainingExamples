/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// 2D shallow water equations (flat bed) with reflective BCs, finite difference space, RK4 time stepping.
// Single precision, MPI decomposition across GPUs, includes performance metric and quick accuracy checks.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(cmd)                                                              \
    do {                                                                             \
        hipError_t e = (cmd);                                                        \
        if (e != hipSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    hipGetErrorString(e));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                            \
    } while (0)
#endif

// -------------------- Global problem size (unchanged) --------------------
constexpr int   NXG     = 8192;     // global interior cells in x
constexpr int   NYG     = 8192;     // global interior cells in y
constexpr float DX      = 1.0f;
constexpr float DY      = 1.0f;
constexpr float G       = 9.81f;
constexpr float CFL     = 0.25f;    // CFL used to set dt from initial wave speed
constexpr int   NSTEPS  = 500;
constexpr float C_VIS   = 0.02f;    // light viscosity

// Set to 1 if your MPI is ROCm/CUDA-aware (recommended). Set to 0 to use host bounce buffers.
#ifndef GPU_AWARE_MPI
#define GPU_AWARE_MPI 1
#endif

constexpr int L1_CACHELINE_BYTES  = 64;
constexpr int L2_CACHELINE_BYTES  = 128;
constexpr int L1_CACHELINE_FLOATS = L1_CACHELINE_BYTES / sizeof(float);  // 16
constexpr int L2_CACHELINE_FLOATS = L2_CACHELINE_BYTES / sizeof(float);  // 32

// grid index helper (pitch may include padding to satisfy cache-line alignment)
__host__ __device__ inline int IDX(int i, int j, int pitch) { return j * pitch + i; }

// -------------------- Kernels (now parameterized by local nx,ny) --------------------

// Initialize: still water + global-centered Gaussian bump
__global__ void init_gaussian(float* __restrict__ h,
                              float* __restrict__ hu,
                              float* __restrict__ hv,
                              int pitch, int nx, int ny,
                              int y_off_glob, // # of interior rows below this subdomain
                              float h0, float amp,
                              float x0, float y0, float sx, float sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // local interior
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;

    // Global coordinates
    int jg = y_off_glob + j; // global interior index [1..NYG]
    float x = (i - 1) * DX;
    float y = (jg - 1) * DY;

    float dx = (x - x0) / sx;
    float dy = (y - y0) / sy;
    float bump = amp * __expf(-0.5f * (dx*dx + dy*dy));

    int id = IDX(i, j, pitch);
    h[id]  = h0 + bump;
    hu[id] = 0.0f;
    hv[id] = 0.0f;
}

// Reflective BCs: always reflect in X; reflect in Y only at physical ends
__global__ void apply_reflect_bc_x_yphys(float* __restrict__ h,
                                         float* __restrict__ hu,
                                         float* __restrict__ hv,
                                         int pitch, int nx, int ny,
                                         int isBottom, int isTop)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    // Left/Right along j=1..ny
    if (t < ny) {
        int j = t + 1;
        // left ghost i=0 from i=1 (flip hu)
        h [IDX(0,    j, pitch)] = h [IDX(1,    j, pitch)];
        hu[IDX(0,    j, pitch)] = -hu[IDX(1,   j, pitch)];
        hv[IDX(0,    j, pitch)] =  hv[IDX(1,   j, pitch)];
        // right ghost i=nx+1 from i=nx
        h [IDX(nx+1, j, pitch)] = h [IDX(nx,   j, pitch)];
        hu[IDX(nx+1, j, pitch)] = -hu[IDX(nx,  j, pitch)];
        hv[IDX(nx+1, j, pitch)] =  hv[IDX(nx,  j, pitch)];
    }

    // Bottom/Top along i=1..nx only at physical boundaries
    if (t < nx) {
        int i = t + 1;
        if (isBottom) {
            // bottom ghost j=0 from j=1 (flip hv)
            h [IDX(i, 0,      pitch)] = h [IDX(i, 1,      pitch)];
            hu[IDX(i, 0,      pitch)] = hu[IDX(i, 1,      pitch)];
            hv[IDX(i, 0,      pitch)] = -hv[IDX(i, 1,     pitch)];
        }
        if (isTop) {
            // top ghost j=ny+1 from j=ny (flip hv)
            h [IDX(i, ny+1,   pitch)] = h [IDX(i, ny,     pitch)];
            hu[IDX(i, ny+1,   pitch)] = hu[IDX(i, ny,     pitch)];
            hv[IDX(i, ny+1,   pitch)] = -hv[IDX(i, ny,    pitch)];
        }
    }

    // Corners only if physical
    if (t == 0) {
        if (isBottom) {
            // bottom-left
            h [IDX(0,    0,    pitch)] = h [IDX(1,    1,    pitch)];
            hu[IDX(0,    0,    pitch)] = -hu[IDX(1,   1,    pitch)];
            hv[IDX(0,    0,    pitch)] = -hv[IDX(1,   1,    pitch)];
            // bottom-right
            h [IDX(nx+1, 0,    pitch)] = h [IDX(nx,   1,    pitch)];
            hu[IDX(nx+1, 0,    pitch)] = -hu[IDX(nx,  1,    pitch)];
            hv[IDX(nx+1, 0,    pitch)] = -hv[IDX(nx,  1,    pitch)];
        }
        if (isTop) {
            // top-left
            h [IDX(0,    ny+1, pitch)] = h [IDX(1,    ny,   pitch)];
            hu[IDX(0,    ny+1, pitch)] = -hu[IDX(1,   ny,   pitch)];
            hv[IDX(0,    ny+1, pitch)] = -hv[IDX(1,   ny,   pitch)];
            // top-right
            h [IDX(nx+1, ny+1, pitch)] = h [IDX(nx,   ny,   pitch)];
            hu[IDX(nx+1, ny+1, pitch)] = -hu[IDX(nx,  ny,   pitch)];
            hv[IDX(nx+1, ny+1, pitch)] = -hv[IDX(nx,  ny,   pitch)];
        }
    }
}

// RHS with central differences + light Laplacian viscosity
__global__ __launch_bounds__(256,1) void compute_rhs_range(const float* __restrict__ h,
                            const float* __restrict__ hu,
                            const float* __restrict__ hv,
                            float* __restrict__ dhdt,
                            float* __restrict__ dhudt,
                            float* __restrict__ dhvdt,
                            int pitch, int nx, int ny,
                            int j_start, int j_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + j_start;
    if (i > nx || j > j_end) return;

    const int id = IDX(i, j, pitch);
    const int ip = id + 1;
    const int im = id - 1;
    const int jp = id + pitch;
    const int jm = id - pitch;

    // AGGRESSIVE GPU MEMORY PREFETCHING STRATEGY:
    // 1. Vectorized loads (float4) for contiguous x-direction data
    // 2. Load ALL data early to maximize memory-level parallelism (MLP)
    // 3. Explicitly load ahead to prime the cache for future iterations
    // 4. Group loads by array type for better coalescing within warps

    // PHASE 1: Issue ALL memory loads FIRST (maximizes MLP and hides latency)
    // Load contiguous x-neighbors as float4 (im, id, ip, and one extra)
    const float4 h_x_vec = *reinterpret_cast<const float4*>(&h[im]);
    const float h_jm = h[jm];  // y-neighbors are not contiguous, load separately
    const float h_jp = h[jp];

    const float4 hu_x_vec = *reinterpret_cast<const float4*>(&hu[im]);
    const float hu_jm = hu[jm];
    const float hu_jp = hu[jp];

    const float4 hv_x_vec = *reinterpret_cast<const float4*>(&hv[im]);
    const float hv_jm = hv[jm];
    const float hv_jp = hv[jp];

    // Prefetch next row data (j+2) to warm cache for future work
    // This is speculative prefetching to hide memory latency
    if (j + 2 <= ny) {
        const int prefetch_row = id + 2*pitch;
        volatile float4 pf_h  = *reinterpret_cast<const float4*>(&h[prefetch_row]);
        volatile float4 pf_hu = *reinterpret_cast<const float4*>(&hu[prefetch_row]);
        volatile float4 pf_hv = *reinterpret_cast<const float4*>(&hv[prefetch_row]);
        (void)pf_h; (void)pf_hu; (void)pf_hv;  // Prevent compiler from optimizing away
    }

    // PHASE 2: Extract values from vectorized loads
    const float h_im = h_x_vec.x;
    const float hi   = h_x_vec.y;
    const float h_ip = h_x_vec.z;

    const float hu_im = hu_x_vec.x;
    const float hui   = hu_x_vec.y;
    const float hu_ip = hu_x_vec.z;

    const float hv_im = hv_x_vec.x;
    const float hvi   = hv_x_vec.y;
    const float hv_ip = hv_x_vec.z;

    // Velocities using one reciprocal per location
    const float eps = 1e-6f;
    const float inv_hi  = 1.0f / fmaxf(hi,  eps);
    const float inv_hip = 1.0f / fmaxf(h_ip, eps);
    const float inv_him = 1.0f / fmaxf(h_im, eps);
    const float inv_hjp = 1.0f / fmaxf(h_jp, eps);
    const float inv_hjm = 1.0f / fmaxf(h_jm, eps);

    const float ui  = hui  * inv_hi;
    const float vi  = hvi  * inv_hi;
    const float u_ip = hu_ip * inv_hip, v_ip = hv_ip * inv_hip;
    const float u_im = hu_im * inv_him, v_im = hv_im * inv_him;
    const float u_jp = hu_jp * inv_hjp, v_jp = hv_jp * inv_hjp;
    const float u_jm = hu_jm * inv_hjm, v_jm = hv_jm * inv_hjm;

    // Central flux derivatives (explicit, avoiding repeated divides)
    const float inv2dx = 0.5f / DX;
    const float inv2dy = 0.5f / DY;

    const float dF1dx = (hu_ip - hu_im) * inv2dx;
    const float F2_ip = fmaf(0.5f*G, h_ip*h_ip, hu_ip * u_ip);
    const float F2_im = fmaf(0.5f*G, h_im*h_im, hu_im * u_im);
    const float dF2dx = (F2_ip - F2_im) * inv2dx;
    const float dF3dx = (hu_ip * v_ip - hu_im * v_im) * inv2dx;

    const float dG1dy = (hv_jp - hv_jm) * inv2dy;
    const float dG2dy = (hu_jp * v_jp - hu_jm * v_jm) * inv2dy;
    const float G3_jp = fmaf(0.5f*G, h_jp*h_jp, hv_jp * v_jp);
    const float G3_jm = fmaf(0.5f*G, h_jm*h_jm, hv_jm * v_jm);
    const float dG3dy = (G3_jp - G3_jm) * inv2dy;

    const float rhs_h  = -(dF1dx + dG1dy);
    const float rhs_hu = -(dF2dx + dG2dy);
    const float rhs_hv = -(dF3dx + dG3dy);

    // Local viscosity term
    const float c  = sqrtf(G * fmaxf(hi, eps)) + fabsf(ui) + fabsf(vi);
    const float nu = C_VIS * c * fmaxf(DX, DY);
    const float inv_dx2 = 1.0f / (DX*DX);
    const float inv_dy2 = 1.0f / (DY*DY);

    const float lap_h  = (h_ip  - 2.f*hi  + h_im)  * inv_dx2 + (h_jp  - 2.f*hi  + h_jm)  * inv_dy2;
    const float lap_hu = (hu_ip - 2.f*hui + hu_im) * inv_dx2 + (hu_jp - 2.f*hui + hu_jm) * inv_dy2;
    const float lap_hv = (hv_ip - 2.f*hvi + hv_im) * inv_dx2 + (hv_jp - 2.f*hvi + hv_jm) * inv_dy2;

    dhdt [id] = rhs_h  + nu * lap_h;
    dhudt[id] = rhs_hu + nu * lap_hu;
    dhvdt[id] = rhs_hv + nu * lap_hv;
}

// ytmp = y + a*dt*k
__global__ void update_stage(float* __restrict__ h_tmp,
                             float* __restrict__ hu_tmp,
                             float* __restrict__ hv_tmp,
                             const float* __restrict__ h,
                             const float* __restrict__ hu,
                             const float* __restrict__ hv,
                             const float* __restrict__ k_h,
                             const float* __restrict__ k_hu,
                             const float* __restrict__ k_hv,
                             float a_dt, int pitch, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = IDX(i, j, pitch);
    h_tmp [id] = h [id] + a_dt * k_h [id];
    hu_tmp[id] = hu[id] + a_dt * k_hu[id];
    hv_tmp[id] = hv[id] + a_dt * k_hv[id];
}

// y = y + dt*(k1 + 2k2 + 2k3 + k4)/6
__global__ void final_update(float* __restrict__ h,
                             float* __restrict__ hu,
                             float* __restrict__ hv,
                             const float* __restrict__ k1_h,
                             const float* __restrict__ k1_hu,
                             const float* __restrict__ k1_hv,
                             const float* __restrict__ k2_h,
                             const float* __restrict__ k2_hu,
                             const float* __restrict__ k2_hv,
                             const float* __restrict__ k3_h,
                             const float* __restrict__ k3_hu,
                             const float* __restrict__ k3_hv,
                             const float* __restrict__ k4_h,
                             const float* __restrict__ k4_hu,
                             const float* __restrict__ k4_hv,
                             float dt, int pitch, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = IDX(i, j, pitch);
    float w_h  = (k1_h [id] + 2.f*k2_h [id] + 2.f*k3_h [id] + k4_h [id]) * (dt/6.f);
    float w_hu = (k1_hu[id] + 2.f*k2_hu[id] + 2.f*k3_hu[id] + k4_hu[id]) * (dt/6.f);
    float w_hv = (k1_hv[id] + 2.f*k2_hv[id] + 2.f*k3_hv[id] + k4_hv[id]) * (dt/6.f);
    h [id] += w_h;  hu[id] += w_hu;  hv[id] += w_hv;
}

// Pack (h,hu,hv) triplets from bottom/top interior rows into contiguous halos
__global__ void pack_y_halos(const float* __restrict__ h,
                             const float* __restrict__ hu,
                             const float* __restrict__ hv,
                             float* __restrict__ send_bottom,
                             float* __restrict__ send_top,
                             int pitch, int nx, int ny,
                             int pack_bottom, int pack_top)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx) return;
    int base = 3 * t;
    if (pack_bottom) {
        int id = IDX(t + 1, 1, pitch);
        send_bottom[base + 0] = h [id];
        send_bottom[base + 1] = hu[id];
        send_bottom[base + 2] = hv[id];
    }
    if (pack_top) {
        int id = IDX(t + 1, ny, pitch);
        send_top[base + 0] = h [id];
        send_top[base + 1] = hu[id];
        send_top[base + 2] = hv[id];
    }
}

// Scatter received halo triplets back into ghost layers
__global__ void unpack_y_halos(float* __restrict__ h,
                               float* __restrict__ hu,
                               float* __restrict__ hv,
                               const float* __restrict__ recv_bottom,
                               const float* __restrict__ recv_top,
                               int pitch, int nx, int ny,
                               int unpack_bottom, int unpack_top)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx) return;
    int base = 3 * t;
    if (unpack_bottom) {
        int id = IDX(t + 1, 0, pitch);
        h [id] = recv_bottom[base + 0];
        hu[id] = recv_bottom[base + 1];
        hv[id] = recv_bottom[base + 2];
    }
    if (unpack_top) {
        int id = IDX(t + 1, ny + 1, pitch);
        h [id] = recv_top[base + 0];
        hu[id] = recv_top[base + 1];
        hv[id] = recv_top[base + 2];
    }
}

// -------------------- Host helpers --------------------

static inline int align_up_int(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static inline float initial_dt(float hmax) {
    float c0 = sqrtf(G * hmax);
    return CFL * fminf(DX, DY) / (c0 + 1e-6f);
}

static inline void reduce_mass_min_local(const std::vector<float>& h_host, int pitch, int nx, int ny,
                                         double& mass, float& hmin)
{
    mass = 0.0; hmin = 1e30f;
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            float hh = h_host[IDX(i, j, pitch)];
            mass += hh;
            hmin = std::min(hmin, hh);
        }
    }
    mass *= (double)DX * (double)DY;
}

struct HaloBuffers {
    float* send_bottom = nullptr;
    float* send_top    = nullptr;
    float* recv_bottom = nullptr;
    float* recv_top    = nullptr;
#if !GPU_AWARE_MPI
    float* host_send_bottom = nullptr;
    float* host_send_top    = nullptr;
    float* host_recv_bottom = nullptr;
    float* host_recv_top    = nullptr;
#endif
    hipStream_t stream = nullptr;
    hipEvent_t  ready  = nullptr;  // orders the pack behind prior default-stream work
    int    halo_count  = 0;
    size_t halo_bytes  = 0;
};

constexpr int TAG_TO_DOWN = 100;
constexpr int TAG_TO_UP   = 101;

static inline dim3 make_grid_2d(int nx, int ny_local, const dim3& block)
{
    int gx = (nx + block.x - 1) / block.x;
    int gy = (ny_local + block.y - 1) / block.y;
    if (gy == 0) gy = 1; // ensure valid launch even for single-row slices
    return dim3(gx, gy, 1);
}

static inline int begin_exchange_y_halos(float* d_h, float* d_hu, float* d_hv,
                                         int pitch, int nx, int ny,
                                         int rank_down, int rank_up, MPI_Comm comm,
                                         HaloBuffers& halo, MPI_Request* reqs)
{
    const int need_down = (rank_down != MPI_PROC_NULL);
    const int need_up   = (rank_up   != MPI_PROC_NULL);
    if (!need_down && !need_up) return 0;

    roctxRangePushA("halo_pack");

    // The pack reads the interior that the default stream just wrote, and the
    // halo stream is non-blocking, so the dependency has to be explicit.
    CHECK_CUDA(hipEventRecord(halo.ready, 0));
    CHECK_CUDA(hipStreamWaitEvent(halo.stream, halo.ready, 0));

    dim3 block(256);
    dim3 grid((nx + block.x - 1) / block.x);
    pack_y_halos<<<grid, block, 0, halo.stream>>>(d_h, d_hu, d_hv,
                                                  halo.send_bottom, halo.send_top,
                                                  pitch, nx, ny,
                                                  need_down, need_up);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipStreamSynchronize(halo.stream));

#if GPU_AWARE_MPI
    float* send_down_ptr = halo.send_bottom;
    float* send_up_ptr   = halo.send_top;
    float* recv_down_ptr = halo.recv_bottom;
    float* recv_up_ptr   = halo.recv_top;
#else
    CHECK_CUDA(hipMemcpyAsync(halo.host_send_bottom, halo.send_bottom, halo.halo_bytes,
                              hipMemcpyDeviceToHost, halo.stream));
    CHECK_CUDA(hipMemcpyAsync(halo.host_send_top, halo.send_top, halo.halo_bytes,
                              hipMemcpyDeviceToHost, halo.stream));
    CHECK_CUDA(hipStreamSynchronize(halo.stream));
    float* send_down_ptr = halo.host_send_bottom;
    float* send_up_ptr   = halo.host_send_top;
    float* recv_down_ptr = halo.host_recv_bottom;
    float* recv_up_ptr   = halo.host_recv_top;
#endif

    roctxRangePop();   // halo_pack

    roctxRangePushA("halo_mpi_post");
    int reqCount = 0;
    if (need_down) {
        MPI_Irecv(recv_down_ptr, halo.halo_count, MPI_FLOAT, rank_down, TAG_TO_UP,
                  comm, &reqs[reqCount++]);
        MPI_Isend(send_down_ptr, halo.halo_count, MPI_FLOAT, rank_down, TAG_TO_DOWN,
                  comm, &reqs[reqCount++]);
    }
    if (need_up) {
        MPI_Irecv(recv_up_ptr, halo.halo_count, MPI_FLOAT, rank_up, TAG_TO_DOWN,
                  comm, &reqs[reqCount++]);
        MPI_Isend(send_up_ptr, halo.halo_count, MPI_FLOAT, rank_up, TAG_TO_UP,
                  comm, &reqs[reqCount++]);
    }
    roctxRangePop();   // halo_mpi_post

    return reqCount;
}

static inline void finish_exchange_y_halos(float* d_h, float* d_hu, float* d_hv,
                                           int pitch, int nx, int ny,
                                           int rank_down, int rank_up,
                                           HaloBuffers& halo,
                                           MPI_Request* reqs, int reqCount)
{
    const int need_down = (rank_down != MPI_PROC_NULL);
    const int need_up   = (rank_up   != MPI_PROC_NULL);
    if (!need_down && !need_up) return;

    roctxRangePushA("halo_mpi_wait");
    MPI_Waitall(reqCount, reqs, MPI_STATUSES_IGNORE);
    roctxRangePop();   // halo_mpi_wait

    roctxRangePushA("halo_unpack");
#if !GPU_AWARE_MPI
    if (need_down) {
        CHECK_CUDA(hipMemcpyAsync(halo.recv_bottom, halo.host_recv_bottom, halo.halo_bytes,
                                  hipMemcpyHostToDevice, halo.stream));
    }
    if (need_up) {
        CHECK_CUDA(hipMemcpyAsync(halo.recv_top, halo.host_recv_top, halo.halo_bytes,
                                  hipMemcpyHostToDevice, halo.stream));
    }
    CHECK_CUDA(hipStreamSynchronize(halo.stream));
#endif

    dim3 block(256);
    dim3 grid((nx + block.x - 1) / block.x);
    unpack_y_halos<<<grid, block, 0, halo.stream>>>(d_h, d_hu, d_hv,
                                                    halo.recv_bottom, halo.recv_top,
                                                    pitch, nx, ny,
                                                    need_down, need_up);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipStreamSynchronize(halo.stream));
    roctxRangePop();   // halo_unpack
}

static inline void synchronize_y_halos(float* d_h, float* d_hu, float* d_hv,
                                       int pitch, int nx, int ny,
                                       int rank_down, int rank_up, MPI_Comm comm,
                                       HaloBuffers& halo)
{
    MPI_Request reqs[4];
    int reqCount = begin_exchange_y_halos(d_h, d_hu, d_hv,
                                          pitch, nx, ny,
                                          rank_down, rank_up, comm,
                                          halo, reqs);
    finish_exchange_y_halos(d_h, d_hu, d_hv,
                            pitch, nx, ny,
                            rank_down, rank_up,
                            halo, reqs, reqCount);
}

static inline void compute_rhs_with_overlap(float* d_h, float* d_hu, float* d_hv,
                                            float* d_dhdt,
                                            float* d_dhudt,
                                            float* d_dhvdt,
                                            int pitch, int nx, int ny,
                                            int rank_down, int rank_up, MPI_Comm comm,
                                            const dim3& block_rhs,
                                            HaloBuffers& halo)
{
    MPI_Request reqs[4];
    int reqCount = begin_exchange_y_halos(d_h, d_hu, d_hv,
                                          pitch, nx, ny,
                                          rank_down, rank_up, comm,
                                          halo, reqs);

    if (ny > 2) {
        roctxRangePushA("rhs_interior");
        int interior_rows = ny - 2;
        dim3 grid_interior = make_grid_2d(nx, interior_rows, block_rhs);
        compute_rhs_range<<<grid_interior, block_rhs>>>(d_h, d_hu, d_hv,
                                                       d_dhdt, d_dhudt, d_dhvdt,
                                                       pitch, nx, ny,
                                                       2, ny - 1);
        CHECK_CUDA(hipGetLastError());
        roctxRangePop();   // rhs_interior
    }

    finish_exchange_y_halos(d_h, d_hu, d_hv,
                            pitch, nx, ny,
                            rank_down, rank_up,
                            halo, reqs, reqCount);

    roctxRangePushA("rhs_edges");
    if (ny >= 1) {
        dim3 grid_edge = make_grid_2d(nx, 1, block_rhs);
        compute_rhs_range<<<grid_edge, block_rhs>>>(d_h, d_hu, d_hv,
                                                    d_dhdt, d_dhudt, d_dhvdt,
                                                    pitch, nx, ny,
                                                    1, 1);
        CHECK_CUDA(hipGetLastError());
    }
    if (ny >= 2) {
        dim3 grid_edge = make_grid_2d(nx, 1, block_rhs);
        compute_rhs_range<<<grid_edge, block_rhs>>>(d_h, d_hu, d_hv,
                                                    d_dhdt, d_dhudt, d_dhvdt,
                                                    pitch, nx, ny,
                                                    ny, ny);
        CHECK_CUDA(hipGetLastError());
    }
    roctxRangePop();   // rhs_edges
}

// -------------------- Main --------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Pick device per rank
    int nd;
    CHECK_CUDA(hipGetDeviceCount(&nd));
    if (nd < 1) {
        if (rank == 0) fprintf(stderr, "No HIP devices found.\n");
        MPI_Abort(comm, 1);
    }
    int dev = rank % nd;
    CHECK_CUDA(hipSetDevice(dev));

    // 1D decomposition in Y
    const int nx = NXG; // not decomposed in X
    int base = NYG / size, rem = NYG % size;
    int ny   = base + (rank < rem ? 1 : 0);
    int y_off_glob = rank * base + (rank < rem ? rank : rem); // #rows below this rank

    // Align pitch so each row begins on a full 128-byte cache line while keeping logical indexing unchanged.
    const int pitch = align_up_int(nx + 2, L1_CACHELINE_FLOATS);
    const size_t nTot  = static_cast<size_t>(pitch) * static_cast<size_t>(ny + 2);
    const size_t bytes = nTot * sizeof(float);

    // Neighbor ranks in Y
    int rank_down = (rank == 0       ? MPI_PROC_NULL : rank - 1);
    int rank_up   = (rank == size-1  ? MPI_PROC_NULL : rank + 1);
    int isBottom  = (rank == 0);
    int isTop     = (rank == size-1);

    // Allocate device arrays
    float *d_h, *d_hu, *d_hv;
    float *d_h_tmp, *d_hu_tmp, *d_hv_tmp;
    float *d_k1_h, *d_k1_hu, *d_k1_hv;
    float *d_k2_h, *d_k2_hu, *d_k2_hv;
    float *d_k3_h, *d_k3_hu, *d_k3_hv;
    float *d_k4_h, *d_k4_hu, *d_k4_hv;

    CHECK_CUDA(hipMalloc(&d_h, bytes));
    CHECK_CUDA(hipMalloc(&d_hu, bytes));
    CHECK_CUDA(hipMalloc(&d_hv, bytes));
    CHECK_CUDA(hipMalloc(&d_h_tmp, bytes));
    CHECK_CUDA(hipMalloc(&d_hu_tmp, bytes));
    CHECK_CUDA(hipMalloc(&d_hv_tmp, bytes));
    CHECK_CUDA(hipMalloc(&d_k1_h, bytes));
    CHECK_CUDA(hipMalloc(&d_k1_hu, bytes));
    CHECK_CUDA(hipMalloc(&d_k1_hv, bytes));
    CHECK_CUDA(hipMalloc(&d_k2_h, bytes));
    CHECK_CUDA(hipMalloc(&d_k2_hu, bytes));
    CHECK_CUDA(hipMalloc(&d_k2_hv, bytes));
    CHECK_CUDA(hipMalloc(&d_k3_h, bytes));
    CHECK_CUDA(hipMalloc(&d_k3_hu, bytes));
    CHECK_CUDA(hipMalloc(&d_k3_hv, bytes));
    CHECK_CUDA(hipMalloc(&d_k4_h, bytes));
    CHECK_CUDA(hipMalloc(&d_k4_hu, bytes));
    CHECK_CUDA(hipMalloc(&d_k4_hv, bytes));

    HaloBuffers halo;
    halo.halo_count = 3 * nx;
    halo.halo_bytes = static_cast<size_t>(halo.halo_count) * sizeof(float);
    CHECK_CUDA(hipMalloc(&halo.send_bottom, halo.halo_bytes));
    CHECK_CUDA(hipMalloc(&halo.send_top,    halo.halo_bytes));
    CHECK_CUDA(hipMalloc(&halo.recv_bottom, halo.halo_bytes));
    CHECK_CUDA(hipMalloc(&halo.recv_top,    halo.halo_bytes));
#if !GPU_AWARE_MPI
    CHECK_CUDA(hipHostMalloc(&halo.host_send_bottom, halo.halo_bytes));
    CHECK_CUDA(hipHostMalloc(&halo.host_send_top,    halo.halo_bytes));
    CHECK_CUDA(hipHostMalloc(&halo.host_recv_bottom, halo.halo_bytes));
    CHECK_CUDA(hipHostMalloc(&halo.host_recv_top,    halo.halo_bytes));
#endif
    CHECK_CUDA(hipStreamCreateWithFlags(&halo.stream, hipStreamNonBlocking));
    CHECK_CUDA(hipEventCreateWithFlags(&halo.ready, hipEventDisableTiming));

    // Launch geometry (local)
    dim3 block(64,4);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
    int nthreads_bc = 256;
    dim3 gridBC((std::max(nx, ny) + (nthreads_bc-1))/nthreads_bc);

    // Init fields (global-centered Gaussian)
    const float h0  = 1.0f;
    const float amp = 0.2f;
    const float x0  = 0.5f * (NXG-1) * DX;
    const float y0  = 0.5f * (NYG-1) * DY;
    const float sx  = 20.0f, sy = 20.0f;

    init_gaussian<<<grid, block>>>(d_h, d_hu, d_hv, pitch, nx, ny, y_off_glob,
                                   h0, amp, x0, y0, sx, sy);
    CHECK_CUDA(hipGetLastError());

    // Set X BCs and physical Y BCs; then exchange halos once to fill Y ghosts
    apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h, d_hu, d_hv, pitch, nx, ny, isBottom, isTop);
    CHECK_CUDA(hipGetLastError());
    synchronize_y_halos(d_h, d_hu, d_hv, pitch, nx, ny, rank_down, rank_up, comm, halo);

    // Host copy for baseline accuracy
    std::vector<float> h0_host(nTot);
    CHECK_CUDA(hipMemcpy(h0_host.data(), d_h, bytes, hipMemcpyDeviceToHost));
    double mass0_local; float hmin0_local;
    reduce_mass_min_local(h0_host, pitch, nx, ny, mass0_local, hmin0_local);

    double mass0 = 0.0, mass1 = 0.0, mass0_sum = 0.0, mass1_sum = 0.0;
    float hmin0 = 0.f, hmin = 0.f, hmin_local = 0.f;
    MPI_Allreduce(&mass0_local, &mass0_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&hmin0_local, &hmin0,     1, MPI_FLOAT,  MPI_MIN, comm);
    mass0 = mass0_sum;

    // Global, conservative dt
    float dt = initial_dt(h0 + amp);
    // Ensure all ranks use the same dt (they already do, but formalize)
    MPI_Bcast(&dt, 1, MPI_FLOAT, 0, comm);

    // -------------------- Time stepping (RK4) --------------------
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    char step_name[24];
    roctxRangePushA("timeloop");
    for (int n = 0; n < NSTEPS; ++n) {
        sprintf(step_name, "step_%d", n);
        roctxRangePushA(step_name);

        // k1: use (d_h, d_hu, d_hv)
        roctxRangePushA("rk4_k1");
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h, d_hu, d_hv, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        compute_rhs_with_overlap(d_h, d_hu, d_hv,
                                 d_k1_h, d_k1_hu, d_k1_hv,
                                 pitch, nx, ny,
                                 rank_down, rank_up, comm,
                                 block, halo);
        roctxRangePop();   // rk4_k1

        // k2: ytmp = y + 0.5*dt*k1
        roctxRangePushA("rk4_k2");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k1_h, d_k1_hu, d_k1_hv, 0.5f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        compute_rhs_with_overlap(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                 d_k2_h, d_k2_hu, d_k2_hv,
                                 pitch, nx, ny,
                                 rank_down, rank_up, comm,
                                 block, halo);
        roctxRangePop();   // rk4_k2

        // k3: ytmp = y + 0.5*dt*k2
        roctxRangePushA("rk4_k3");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k2_h, d_k2_hu, d_k2_hv, 0.5f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        compute_rhs_with_overlap(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                 d_k3_h, d_k3_hu, d_k3_hv,
                                 pitch, nx, ny,
                                 rank_down, rank_up, comm,
                                 block, halo);
        roctxRangePop();   // rk4_k3

        // k4: ytmp = y + dt*k3
        roctxRangePushA("rk4_k4");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k3_h, d_k3_hu, d_k3_hv, 1.0f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        compute_rhs_with_overlap(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                 d_k4_h, d_k4_hu, d_k4_hv,
                                 pitch, nx, ny,
                                 rank_down, rank_up, comm,
                                 block, halo);
        roctxRangePop();   // rk4_k4

        // Final update
        roctxRangePushA("final_update");
        final_update<<<grid, block>>>(d_h, d_hu, d_hv,
                                      d_k1_h, d_k1_hu, d_k1_hv,
                                      d_k2_h, d_k2_hu, d_k2_hv,
                                      d_k3_h, d_k3_hu, d_k3_hv,
                                      d_k4_h, d_k4_hu, d_k4_hv,
                                      dt, pitch, nx, ny);
        roctxRangePop();   // final_update

        roctxRangePop();   // step
    }
    roctxRangePop();   // timeloop

    CHECK_CUDA(hipDeviceSynchronize());
    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    double seconds_local = t1 - t0, seconds = 0.0;
    MPI_Allreduce(&seconds_local, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);

    // -------------------- Accuracy checks --------------------
    std::vector<float> h_host(nTot);
    CHECK_CUDA(hipMemcpy(h_host.data(), d_h, bytes, hipMemcpyDeviceToHost));
    double mass1_local; float hmin_local_after;
    reduce_mass_min_local(h_host, pitch, nx, ny, mass1_local, hmin_local_after);

    MPI_Allreduce(&mass1_local, &mass1_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&hmin_local_after, &hmin, 1, MPI_FLOAT, MPI_MIN, comm);
    mass1 = mass1_sum;

    double rel_mass_err = fabs(mass1 - mass0) / std::max(1e-12, fabs(mass0));

    // -------------------- Performance --------------------
    double cells_global   = (double)NXG * (double)NYG;
    double updates_global = cells_global * 4.0 * (double)NSTEPS; // includes RK4 stages
    double mcups = updates_global / seconds / 1.0e6;

    if (rank == 0) {
        printf("MPI ranks: %d  |  GPUs detected: %d\n", size, nd);
        printf("Domain: %dx%d (global), steps=%d, dt=%.6g\n", NXG, NYG, NSTEPS, dt);
        printf("Elapsed (max over ranks): %.3f s  |  Throughput: %.2f MCUPS\n", seconds, mcups);
        printf("Mass: initial=%.9e, final=%.9e, rel.err=%.3e\n", mass0, mass1, rel_mass_err);
        printf("Min(h) after run: %.6g %s\n", hmin, (hmin < 0.f ? "(NEGATIVE!)" : ""));
    }

    // Clean
    hipFree(d_h);    hipFree(d_hu);   hipFree(d_hv);
    hipFree(d_h_tmp); hipFree(d_hu_tmp); hipFree(d_hv_tmp);
    hipFree(d_k1_h); hipFree(d_k1_hu); hipFree(d_k1_hv);
    hipFree(d_k2_h); hipFree(d_k2_hu); hipFree(d_k2_hv);
    hipFree(d_k3_h); hipFree(d_k3_hu); hipFree(d_k3_hv);
    hipFree(d_k4_h); hipFree(d_k4_hu); hipFree(d_k4_hv);
    hipFree(halo.send_bottom); hipFree(halo.send_top);
    hipFree(halo.recv_bottom); hipFree(halo.recv_top);
#if !GPU_AWARE_MPI
    hipHostFree(halo.host_send_bottom); hipHostFree(halo.host_send_top);
    hipHostFree(halo.host_recv_bottom); hipHostFree(halo.host_recv_top);
#endif
    hipEventDestroy(halo.ready);
    hipStreamDestroy(halo.stream);

    MPI_Finalize();
    return 0;
}
