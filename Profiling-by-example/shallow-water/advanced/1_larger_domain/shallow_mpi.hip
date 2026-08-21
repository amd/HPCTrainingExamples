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

// grid index helper (pitch = nx+2)
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
__global__ void compute_rhs(const float* __restrict__ h,
                            const float* __restrict__ hu,
                            const float* __restrict__ hv,
                            float* __restrict__ dhdt,
                            float* __restrict__ dhudt,
                            float* __restrict__ dhvdt,
                            int pitch, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;

    const int id = IDX(i, j, pitch);
    const int ip = IDX(i+1, j, pitch);
    const int im = IDX(i-1, j, pitch);
    const int jp = IDX(i, j+1, pitch);
    const int jm = IDX(i, j-1, pitch);

    const float hi  = h[id];
    const float hui = hu[id];
    const float hvi = hv[id];

    const float eps = 1e-6f;
    const float ui  = hui / fmaxf(hi, eps);
    const float vi  = hvi / fmaxf(hi, eps);

    float h_ip = h[ip], hu_ip = hu[ip], hv_ip = hv[ip];
    float h_im = h[im], hu_im = hu[im], hv_im = hv[im];
    float h_jp = h[jp], hu_jp = hu[jp], hv_jp = hv[jp];
    float h_jm = h[jm], hu_jm = hu[jm], hv_jm = hv[jm];

    auto inv = [&](float H, float HU, float HV, float& U, float& V) {
        float He = fmaxf(H, eps);
        U = HU / He;
        V = HV / He;
    };
    float u_ip, v_ip, u_im, v_im, u_jp, v_jp, u_jm, v_jm;
    inv(h_ip, hu_ip, hv_ip, u_ip, v_ip);
    inv(h_im, hu_im, hv_im, u_im, v_im);
    inv(h_jp, hu_jp, hv_jp, u_jp, v_jp);
    inv(h_jm, hu_jm, hv_jm, u_jm, v_jm);

    auto fx1 = [&](float H, float HU) { return HU; };
    auto fx2 = [&](float H, float HU) { float He=fmaxf(H,eps); float u=HU/He; return HU*u + 0.5f*G*H*H; };
    auto fx3 = [&](float H, float HU, float HV) { float He=fmaxf(H,eps); float u=HU/He; float v=HV/He; return HU*v; };

    auto fy1 = [&](float H, float HV) { return HV; };
    auto fy2 = [&](float H, float HU, float HV) { float He=fmaxf(H,eps); float u=HU/He; float v=HV/He; return HU*v; };
    auto fy3 = [&](float H, float HV) { float He=fmaxf(H,eps); float v=HV/He; return HV*v + 0.5f*G*H*H; };

    const float inv2dx = 0.5f / DX;
    const float inv2dy = 0.5f / DY;

    float dF1dx = (fx1(h_ip, hu_ip) - fx1(h_im, hu_im)) * inv2dx;
    float dF2dx = (fx2(h_ip, hu_ip) - fx2(h_im, hu_im)) * inv2dx;
    float dF3dx = (fx3(h_ip, hu_ip, hv_ip) - fx3(h_im, hu_im, hv_im)) * inv2dx;

    float dG1dy = (fy1(h_jp, hv_jp) - fy1(h_jm, hv_jm)) * inv2dy;
    float dG2dy = (fy2(h_jp, hu_jp, hv_jp) - fy2(h_jm, hu_jm, hv_jm)) * inv2dy;
    float dG3dy = (fy3(h_jp, hv_jp) - fy3(h_jm, hv_jm)) * inv2dy;

    float rhs_h  = -(dF1dx + dG1dy);
    float rhs_hu = -(dF2dx + dG2dy);
    float rhs_hv = -(dF3dx + dG3dy);

    float c  = sqrtf(G * fmaxf(hi, eps)) + fabsf(ui) + fabsf(vi);
    float nu = C_VIS * c * fmaxf(DX, DY);

    float lap_h  = (h[ip]  - 2.f*hi  + h[im])  / (DX*DX) + (h[jp]  - 2.f*hi  + h[jm])  / (DY*DY);
    float lap_hu = (hu[ip] - 2.f*hui + hu[im]) / (DX*DX) + (hu[jp] - 2.f*hui + hu[jm]) / (DY*DY);
    float lap_hv = (hv[ip] - 2.f*hvi + hv[im]) / (DX*DX) + (hv[jp] - 2.f*hvi + hv[jm]) / (DY*DY);

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

// -------------------- Host helpers --------------------

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

// Halo exchange for one set of arrays (h,hu,hv) in Y (1 ghost layer)
static inline void exchange_y_halos(float* d_h, float* d_hu, float* d_hv,
                                    int pitch, int nx, int ny,
                                    int rank_down, int rank_up, MPI_Comm comm)
{
    roctxRangePushA("halo_exchange");

    // Ensure device writes are visible
    CHECK_CUDA(hipDeviceSynchronize());

    const size_t row_bytes = nx * sizeof(float);
    float* send_bot_h = d_h + IDX(1,       1,    pitch);
    float* recv_bot_h = d_h + IDX(1,       0,    pitch);
    float* send_top_h = d_h + IDX(1,       ny,   pitch);
    float* recv_top_h = d_h + IDX(1,       ny+1, pitch);

    float* send_bot_hu = d_hu + IDX(1,       1,    pitch);
    float* recv_bot_hu = d_hu + IDX(1,       0,    pitch);
    float* send_top_hu = d_hu + IDX(1,       ny,   pitch);
    float* recv_top_hu = d_hu + IDX(1,       ny+1, pitch);

    float* send_bot_hv = d_hv + IDX(1,       1,    pitch);
    float* recv_bot_hv = d_hv + IDX(1,       0,    pitch);
    float* send_top_hv = d_hv + IDX(1,       ny,   pitch);
    float* recv_top_hv = d_hv + IDX(1,       ny+1, pitch);

#if GPU_AWARE_MPI
    roctxRangePushA("halo_mpi");
    // bottom neighbor exchange (send our j=1; recv into j=0)
    if (rank_down != MPI_PROC_NULL) {
        MPI_Sendrecv(send_bot_h, nx, MPI_FLOAT, rank_down, 100,
                     recv_bot_h, nx, MPI_FLOAT, rank_down, 200, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot_hu, nx, MPI_FLOAT, rank_down, 101,
                     recv_bot_hu, nx, MPI_FLOAT, rank_down, 201, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot_hv, nx, MPI_FLOAT, rank_down, 102,
                     recv_bot_hv, nx, MPI_FLOAT, rank_down, 202, comm, MPI_STATUS_IGNORE);
    }
    // top neighbor exchange (send our j=ny; recv into j=ny+1)
    if (rank_up != MPI_PROC_NULL) {
        MPI_Sendrecv(send_top_h, nx, MPI_FLOAT, rank_up, 200,
                     recv_top_h, nx, MPI_FLOAT, rank_up, 100, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_top_hu, nx, MPI_FLOAT, rank_up, 201,
                     recv_top_hu, nx, MPI_FLOAT, rank_up, 101, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_top_hv, nx, MPI_FLOAT, rank_up, 202,
                     recv_top_hv, nx, MPI_FLOAT, rank_up, 102, comm, MPI_STATUS_IGNORE);
    }
    roctxRangePop();   // halo_mpi
#else
    // Host bounce buffers (pinned)
    float *hb, *ht, *hub, *hut, *hvb, *hvt;
    CHECK_CUDA(hipHostMalloc(&hb,  row_bytes));
    CHECK_CUDA(hipHostMalloc(&ht,  row_bytes));
    CHECK_CUDA(hipHostMalloc(&hub, row_bytes));
    CHECK_CUDA(hipHostMalloc(&hut, row_bytes));
    CHECK_CUDA(hipHostMalloc(&hvb, row_bytes));
    CHECK_CUDA(hipHostMalloc(&hvt, row_bytes));

    // Copy device->host for sends
    CHECK_CUDA(hipMemcpy(hb,  send_bot_h,  row_bytes, hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(ht,  send_top_h,  row_bytes, hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(hub, send_bot_hu, row_bytes, hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(hut, send_top_hu, row_bytes, hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(hvb, send_bot_hv, row_bytes, hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(hvt, send_top_hv, row_bytes, hipMemcpyDeviceToHost));

    roctxRangePushA("halo_mpi");
    if (rank_down != MPI_PROC_NULL) {
        MPI_Sendrecv(hb,  nx, MPI_FLOAT, rank_down, 100,
                     hb,  nx, MPI_FLOAT, rank_down, 200, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hub, nx, MPI_FLOAT, rank_down, 101,
                     hub, nx, MPI_FLOAT, rank_down, 201, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hvb, nx, MPI_FLOAT, rank_down, 102,
                     hvb, nx, MPI_FLOAT, rank_down, 202, comm, MPI_STATUS_IGNORE);
        // recv buffers overwrite hb/hub/hvb; copy to device ghosts
        CHECK_CUDA(hipMemcpy(recv_bot_h,  hb,  row_bytes, hipMemcpyHostToDevice));
        CHECK_CUDA(hipMemcpy(recv_bot_hu, hub, row_bytes, hipMemcpyHostToDevice));
        CHECK_CUDA(hipMemcpy(recv_bot_hv, hvb, row_bytes, hipMemcpyHostToDevice));
    }
    if (rank_up != MPI_PROC_NULL) {
        MPI_Sendrecv(ht,  nx, MPI_FLOAT, rank_up, 200,
                     ht,  nx, MPI_FLOAT, rank_up, 100, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hut, nx, MPI_FLOAT, rank_up, 201,
                     hut, nx, MPI_FLOAT, rank_up, 101, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hvt, nx, MPI_FLOAT, rank_up, 202,
                     hvt, nx, MPI_FLOAT, rank_up, 102, comm, MPI_STATUS_IGNORE);
        CHECK_CUDA(hipMemcpy(recv_top_h,  ht,  row_bytes, hipMemcpyHostToDevice));
        CHECK_CUDA(hipMemcpy(recv_top_hu, hut, row_bytes, hipMemcpyHostToDevice));
        CHECK_CUDA(hipMemcpy(recv_top_hv, hvt, row_bytes, hipMemcpyHostToDevice));
    }
    roctxRangePop();   // halo_mpi

    hipHostFree(hb); hipHostFree(ht); hipHostFree(hub);
    hipHostFree(hut); hipHostFree(hvb); hipHostFree(hvt);
#endif

    roctxRangePop();   // halo_exchange
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

    const int pitch = nx + 2;
    const int nTot  = (nx + 2) * (ny + 2);
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

    // Launch geometry (local)
    dim3 block(16,16);
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
    exchange_y_halos(d_h, d_hu, d_hv, pitch, nx, ny, rank_down, rank_up, comm);

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
        exchange_y_halos(d_h, d_hu, d_hv, pitch, nx, ny, rank_down, rank_up, comm);
        compute_rhs<<<grid, block>>>(d_h, d_hu, d_hv, d_k1_h, d_k1_hu, d_k1_hv, pitch, nx, ny);
        roctxRangePop();   // rk4_k1

        // k2: ytmp = y + 0.5*dt*k1
        roctxRangePushA("rk4_k2");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k1_h, d_k1_hu, d_k1_hv, 0.5f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        exchange_y_halos(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, rank_down, rank_up, comm);
        compute_rhs<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k2_h, d_k2_hu, d_k2_hv, pitch, nx, ny);
        roctxRangePop();   // rk4_k2

        // k3: ytmp = y + 0.5*dt*k2
        roctxRangePushA("rk4_k3");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k2_h, d_k2_hu, d_k2_hv, 0.5f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        exchange_y_halos(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, rank_down, rank_up, comm);
        compute_rhs<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k3_h, d_k3_hu, d_k3_hv, pitch, nx, ny);
        roctxRangePop();   // rk4_k3

        // k4: ytmp = y + dt*k3
        roctxRangePushA("rk4_k4");
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_h, d_hu, d_hv,
                                      d_k3_h, d_k3_hu, d_k3_hv, 1.0f*dt, pitch, nx, ny);
        apply_reflect_bc_x_yphys<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, isBottom, isTop);
        CHECK_CUDA(hipGetLastError());
        exchange_y_halos(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch, nx, ny, rank_down, rank_up, comm);
        compute_rhs<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k4_h, d_k4_hu, d_k4_hv, pitch, nx, ny);
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

    MPI_Finalize();
    return 0;
}
