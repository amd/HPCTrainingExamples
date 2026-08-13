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
// Single precision, simple kernels, includes performance metric and quick accuracy checks.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(cmd)                                                           \
    do {                                                                          \
        hipError_t e = (cmd);                                                    \
        if (e != hipSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    hipGetErrorString(e));                                       \
            exit(1);                                                              \
        }                                                                         \
    } while (0)
#endif

// -------------------- Problem setup --------------------
constexpr int NX = 2048;            // interior cells in x
constexpr int NY = 2048;            // interior cells in y
constexpr float DX = 1.0f;
constexpr float DY = 1.0f;
constexpr float G  = 9.81f;
constexpr float CFL = 0.25f;       // safe CFL (dt computed from initial wave speed)
constexpr int NSTEPS = 500;        // number of time steps

// light numerical viscosity coefficient (scaled by local wave speed)
constexpr float C_VIS = 0.02f;

// grid (with 1-cell ghost ring for reflective BCs)
__host__ __device__ inline int IDX(int i, int j, int pitch) {
    // i in [0..NX+1], j in [0..NY+1]
    return j * pitch + i;
}

// -------------------- Kernels --------------------

// Initialize: still water + Gaussian bump in h
__global__ void init_gaussian(float* __restrict__ h,
                              float* __restrict__ hu,
                              float* __restrict__ hv,
                              int pitch, float h0,
                              float amp, float x0, float y0, float sx, float sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // interior
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > NX || j > NY) return;

    float x = (i - 1) * DX;
    float y = (j - 1) * DY;

    float dx = (x - x0) / sx;
    float dy = (y - y0) / sy;
    float bump = amp * expf(-0.5f * (dx*dx + dy*dy));

    int id = IDX(i, j, pitch);
    h[id]  = h0 + bump;
    hu[id] = 0.0f;  // at rest initially
    hv[id] = 0.0f;
}

// Reflective boundary conditions on 1-cell ghost ring
__global__ void apply_reflect_bc(float* __restrict__ h,
                                 float* __restrict__ hu,
                                 float* __restrict__ hv,
                                 int pitch)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    // Left/Right along j=1..NY
    if (t <= NY) {
        int j = t + 1;

        // left ghost (i=0) mirrors interior i=1, flip normal momentum hu
        h[IDX(0, j, pitch)]   = h[IDX(1, j, pitch)];
        hu[IDX(0, j, pitch)]  = -hu[IDX(1, j, pitch)];
        hv[IDX(0, j, pitch)]  =  hv[IDX(1, j, pitch)];

        // right ghost (i=NX+1) mirrors interior i=NX
        h[IDX(NX+1, j, pitch)]  = h[IDX(NX, j, pitch)];
        hu[IDX(NX+1, j, pitch)] = -hu[IDX(NX, j, pitch)];
        hv[IDX(NX+1, j, pitch)] =  hv[IDX(NX, j, pitch)];
    }

    // Bottom/Top along i=1..NX
    if (t <= NX) {
        int i = t + 1;

        // bottom ghost (j=0) mirrors j=1, flip normal momentum hv
        h[IDX(i, 0, pitch)]   = h[IDX(i, 1, pitch)];
        hu[IDX(i, 0, pitch)]  =  hu[IDX(i, 1, pitch)];
        hv[IDX(i, 0, pitch)]  = -hv[IDX(i, 1, pitch)];

        // top ghost (j=NY+1) mirrors j=NY
        h[IDX(i, NY+1, pitch)]   = h[IDX(i, NY, pitch)];
        hu[IDX(i, NY+1, pitch)]  =  hu[IDX(i, NY, pitch)];
        hv[IDX(i, NY+1, pitch)]  = -hv[IDX(i, NY, pitch)];
    }

    // Corners (use nearest interior corner)
    if (t == 0) {
        // bottom-left
        h[IDX(0, 0, pitch)]   = h[IDX(1, 1, pitch)];
        hu[IDX(0, 0, pitch)]  = -hu[IDX(1, 1, pitch)];
        hv[IDX(0, 0, pitch)]  = -hv[IDX(1, 1, pitch)];
        // bottom-right
        h[IDX(NX+1, 0, pitch)]   = h[IDX(NX, 1, pitch)];
        hu[IDX(NX+1, 0, pitch)]  = -hu[IDX(NX, 1, pitch)];
        hv[IDX(NX+1, 0, pitch)]  = -hv[IDX(NX, 1, pitch)];
        // top-left
        h[IDX(0, NY+1, pitch)]   = h[IDX(1, NY, pitch)];
        hu[IDX(0, NY+1, pitch)]  = -hu[IDX(1, NY, pitch)];
        hv[IDX(0, NY+1, pitch)]  = -hv[IDX(1, NY, pitch)];
        // top-right
        h[IDX(NX+1, NY+1, pitch)]   = h[IDX(NX, NY, pitch)];
        hu[IDX(NX+1, NY+1, pitch)]  = -hu[IDX(NX, NY, pitch)];
        hv[IDX(NX+1, NY+1, pitch)]  = -hv[IDX(NX, NY, pitch)];
    }
}

__device__ __forceinline__
float fx2_val(float H, float HU, float eps) {
    float He = fmaxf(H, eps);
    float u  = HU / He;
    return HU * u + 0.5f * G * H * H; // HU^2/He + 0.5 g H^2
}

__device__ __forceinline__
float fy3_val(float H, float HV, float eps) {
    float He = fmaxf(H, eps);
    float v  = HV / He;
    return HV * v + 0.5f * G * H * H; // HV^2/He + 0.5 g H^2
}

__device__ __forceinline__
void compute_cell(int id,
                  float Him, float Hc,  float Hip,
                  float HUim,float HUc, float HUip,
                  float HVim,float HVc, float HVip,
                  float Hjm, float Hjp,
                  float HUjm,float HUjp,
                  float HVjm,float HVjp,
                  float eps,  float inv2dx, float inv2dy,
                  float invdx2, float invdy2, float hxy,
                  float* __restrict__ dhdt,
                  float* __restrict__ dhudt,
                  float* __restrict__ dhvdt)
{
    float He = fmaxf(Hc, eps);
    float ui = HUc / He;
    float vi = HVc / He;

    // x-flux divergence
    float dF1dx  = (HUip - HUim) * inv2dx;
    float dF2dx  = (fx2_val(Hip, HUip, eps) - fx2_val(Him, HUim, eps)) * inv2dx;
    float vim    = HVim / fmaxf(Him, eps);
    float vip    = HVip / fmaxf(Hip, eps);
    float dF3dx  = (HUip * vip - HUim * vim) * inv2dx;

    // y-flux divergence
    float dG1dy  = (HVjp - HVjm) * inv2dy;
    float vjm    = HVjm / fmaxf(Hjm, eps);
    float vjp    = HVjp / fmaxf(Hjp, eps);
    float dG2dy  = (HUjp * vjp - HUjm * vjm) * inv2dy;
    float dG3dy  = (fy3_val(Hjp, HVjp, eps) - fy3_val(Hjm, HVjm, eps)) * inv2dy;

    // base RHS
    float rhs_h  = -(dF1dx + dG1dy);
    float rhs_hu = -(dF2dx + dG2dy);
    float rhs_hv = -(dF3dx + dG3dy);

    // light Laplacian viscosity
    float c  = sqrtf(G * He) + fabsf(ui) + fabsf(vi);
    float nu = C_VIS * c * hxy;

    float lap_h  = (Hip  - 2.f*Hc  + Him)  * invdx2 + (Hjp  - 2.f*Hc  + Hjm)  * invdy2;
    float lap_hu = (HUip - 2.f*HUc + HUim) * invdx2 + (HUjp - 2.f*HUc + HUjm) * invdy2;
    float lap_hv = (HVip - 2.f*HVc + HVim) * invdx2 + (HVjp - 2.f*HVc + HVjm) * invdy2;

    dhdt[id]  = rhs_h  + nu * lap_h;
    dhudt[id] = rhs_hu + nu * lap_hu;
    dhvdt[id] = rhs_hv + nu * lap_hv;
}


__global__ __launch_bounds__(256,1) void compute_rhs_vec4x4(const float* __restrict__ h,
							    const float* __restrict__ hu,
							    const float* __restrict__ hv,
							    float* __restrict__ dhdt,
							    float* __restrict__ dhudt,
							    float* __restrict__ dhvdt,
							    int pitch)

{
    const int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // interior y
    if (j > NY) return;

    const int t  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = 1 + 4 * t;                                // 4 cells/thread
    if (i0 > NX) return;

    const float eps    = 1e-6f;
    const float inv2dx = 0.5f / DX;
    const float inv2dy = 0.5f / DY;
    const float invdx2 = 1.f / (DX * DX);
    const float invdy2 = 1.f / (DY * DY);
    const float hxy    = fmaxf(DX, DY);

    // Row bases (element indices)
    const int row    = IDX(0, j,   pitch);
    const int row_jm = IDX(0, j-1, pitch);
    const int row_jp = IDX(0, j+1, pitch);

    const float* __restrict__ hR   = h  + row;
    const float* __restrict__ huR  = hu + row;
    const float* __restrict__ hvR  = hv + row;

    // We try to load two float4 windows on the center row:
    // A: [i0-1, i0, i0+1, i0+2], B: [i0+2, i0+3, i0+4, i0+5]
    const bool haveA = (i0 - 1 >= 0) && (i0 + 2 <= NX + 1);
    const bool haveB = (i0 + 5 <= NX + 1);

    float4 H0, HU0, HV0, H1, HU1, HV1;
    if (haveA) {
        H0  = *reinterpret_cast<const float4*>(hR  + (i0 - 1));
        HU0 = *reinterpret_cast<const float4*>(huR + (i0 - 1));
        HV0 = *reinterpret_cast<const float4*>(hvR + (i0 - 1));
    }
    if (haveB) {
        H1  = *reinterpret_cast<const float4*>(hR  + (i0 + 2));
        HU1 = *reinterpret_cast<const float4*>(huR + (i0 + 2));
        HV1 = *reinterpret_cast<const float4*>(hvR + (i0 + 2));
    }

    // Number of valid interior cells left in this 4-wide tile
    int nCells = NX - i0 + 1;
    if (nCells > 4) nCells = 4;

    #pragma unroll
    for (int k = 0; k < nCells; ++k) {
        const int i  = i0 + k;
        const int id = IDX(i,   j, pitch);

        // Scalar vertical neighbors at (i, j±1): robust and cheap
        const int jm = IDX(i, j-1, pitch);
        const int jp = IDX(i, j+1, pitch);
        float Hjm   = h[jm],   Hjp   = h[jp];
        float HUjm  = hu[jm],  HUjp  = hu[jp];
        float HVjm  = hv[jm],  HVjp  = hv[jp];

        // Center row X-neighbor triplets for this k
        // Prefer vector values when they are fully available; otherwise load scalar.
        float Him, Hc, Hip, HUim, HUc, HUip, HVim, HVc, HVip;

        bool useA = haveA && (k <= 1);              // k=0,1 entirely in window A
        bool useB = haveA && haveB && (k >= 2);     // k=2,3 need A+B

        if (useA && k == 0) {
            Him  = H0.x; Hc  = H0.y; Hip  = H0.z;
            HUim = HU0.x; HUc = HU0.y; HUip = HU0.z;
            HVim = HV0.x; HVc = HV0.y; HVip = HV0.z;
        } else if (useA && k == 1) {
            Him  = H0.y; Hc  = H0.z; Hip  = H0.w;
            HUim = HU0.y; HUc = HU0.z; HUip = HU0.w;
            HVim = HV0.y; HVc = HV0.z; HVip = HV0.w;
        } else if (useB && k == 2) {
            Him  = H0.z; Hc  = H1.x; Hip  = H1.y;
            HUim = HU0.z; HUc = HU1.x; HUip = HU1.y;
            HVim = HV0.z; HVc = HV1.x; HVip = HV1.y;
        } else if (useB && k == 3) {
            Him  = H1.x; Hc  = H1.y; Hip  = H1.z;
            HUim = HU1.x; HUc = HU1.y; HUip = HU1.z;
            HVim = HV1.x; HVc = HV1.y; HVip = HV1.z;
        } else {
            // Scalar fallback (edge-safe)
            const int im = IDX(i-1, j, pitch);
            const int ip = IDX(i+1, j, pitch);
            Him  = h[im];  Hc  = h[id];  Hip  = h[ip];
            HUim = hu[im]; HUc = hu[id]; HUip = hu[ip];
            HVim = hv[im]; HVc = hv[id]; HVip = hv[ip];
        }

        compute_cell(id,
                     Him, Hc,  Hip,
                     HUim,HUc, HUip,
                     HVim,HVc, HVip,
                     Hjm, Hjp,
                     HUjm, HUjp,
                     HVjm, HVjp,
                     eps, inv2dx, inv2dy, invdx2, invdy2, hxy,
                     dhdt, dhudt, dhvdt);
    }
}

// Compute RHS (d/dt of conserved variables) using finite differences (central) + light Laplacian viscosity.
// U = [h, hu, hv]; Fluxes:
// Fx = [hu, hu^2/h + 1/2 g h^2, hu*hv/h]
// Fy = [hv, hu*hv/h, hv^2/h + 1/2 g h^2]
__global__ __launch_bounds__(256,1) void compute_rhs(const float* __restrict__ h,
                            const float* __restrict__ hu,
                            const float* __restrict__ hv,
                            float* __restrict__ dhdt,
                            float* __restrict__ dhudt,
                            float* __restrict__ dhvdt,
                            int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // interior
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > NX || j > NY) return;

    const int id    = IDX(i, j, pitch);
    const int ip    = IDX(i+1, j, pitch);
    const int im    = IDX(i-1, j, pitch);
    const int jp    = IDX(i, j+1, pitch);
    const int jm    = IDX(i, j-1, pitch);

    const float hi  = h[id];
    const float hui = hu[id];
    const float hvi = hv[id];

    const float eps = 1e-6f;
    const float ui  = hui / fmaxf(hi, eps);
    const float vi  = hvi / fmaxf(hi, eps);

    // Fluxes at neighbor centers
    // X-direction neighbors
    float h_ip = h[ip], hu_ip = hu[ip], hv_ip = hv[ip];
    float h_im = h[im], hu_im = hu[im], hv_im = hv[im];

    // Y-direction neighbors
    float h_jp = h[jp], hu_jp = hu[jp], hv_jp = hv[jp];
    float h_jm = h[jm], hu_jm = hu[jm], hv_jm = hv[jm];

    // Conserved-to-primitive at neighbors
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

    // Flux vectors (components) at cell centers
    auto fx1 = [&](float H, float HU) { return HU; };
    auto fx2 = [&](float H, float HU) { float He=fmaxf(H,eps); float u=HU/He; return HU*u + 0.5f*G*H*H; };
    auto fx3 = [&](float H, float HU, float HV) { float He=fmaxf(H,eps); float u=HU/He; float v=HV/He; return HU*v; };

    auto fy1 = [&](float H, float HV) { return HV; };
    auto fy2 = [&](float H, float HU, float HV) { float He=fmaxf(H,eps); float u=HU/He; float v=HV/He; return HU*v; };
    auto fy3 = [&](float H, float HV) { float He=fmaxf(H,eps); float v=HV/He; return HV*v + 0.5f*G*H*H; };

    // Central differences of flux divergence
    const float inv2dx = 0.5f / DX;
    const float inv2dy = 0.5f / DY;

    float dF1dx = (fx1(h_ip, hu_ip) - fx1(h_im, hu_im)) * inv2dx;
    float dF2dx = (fx2(h_ip, hu_ip) - fx2(h_im, hu_im)) * inv2dx;
    float dF3dx = (fx3(h_ip, hu_ip, hv_ip) - fx3(h_im, hu_im, hv_im)) * inv2dx;

    float dG1dy = (fy1(h_jp, hv_jp) - fy1(h_jm, hv_jm)) * inv2dy;
    float dG2dy = (fy2(h_jp, hu_jp, hv_jp) - fy2(h_jm, hu_jm, hv_jm)) * inv2dy;
    float dG3dy = (fy3(h_jp, hv_jp) - fy3(h_jm, hv_jm)) * inv2dy;

    // Base RHS = -div(F, G)
    float rhs_h  = -(dF1dx + dG1dy);
    float rhs_hu = -(dF2dx + dG2dy);
    float rhs_hv = -(dF3dx + dG3dy);

    // Light Laplacian viscosity for robustness
    // nu ~ C_VIS * local wave speed * max(dx,dy)
    float c = sqrtf(G * fmaxf(hi, eps)) + fabsf(ui) + fabsf(vi);
    float nu = C_VIS * c * fmaxf(DX, DY);

    float lap_h  = (h[ip]  - 2.f*hi  + h[im])  / (DX*DX) + (h[jp]  - 2.f*hi  + h[jm])  / (DY*DY);
    float lap_hu = (hu[ip] - 2.f*hui + hu[im]) / (DX*DX) + (hu[jp] - 2.f*hui + hu[jm]) / (DY*DY);
    float lap_hv = (hv[ip] - 2.f*hvi + hv[im]) / (DX*DX) + (hv[jp] - 2.f*hvi + hv[jm]) / (DY*DY);

    rhs_h  += nu * lap_h;
    rhs_hu += nu * lap_hu;
    rhs_hv += nu * lap_hv;

    dhdt[id]  = rhs_h;
    dhudt[id] = rhs_hu;
    dhvdt[id] = rhs_hv;
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
                             float a_dt, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > NX || j > NY) return;

    int id = IDX(i, j, pitch);
    h_tmp[id]  = h[id]  + a_dt * k_h[id];
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
                             float dt, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > NX || j > NY) return;

    int id = IDX(i, j, pitch);

    float w_h  = (k1_h[id]  + 2.f*k2_h[id]  + 2.f*k3_h[id]  + k4_h[id])  * (dt/6.f);
    float w_hu = (k1_hu[id] + 2.f*k2_hu[id] + 2.f*k3_hu[id] + k4_hu[id]) * (dt/6.f);
    float w_hv = (k1_hv[id] + 2.f*k2_hv[id] + 2.f*k3_hv[id] + k4_hv[id]) * (dt/6.f);

    h[id]  += w_h;
    hu[id] += w_hu;
    hv[id] += w_hv;
}

// -------------------- Host helpers --------------------

float initial_dt(float h0)
{
    // estimate from initial max wave speed (still water + bump)
    float c0 = sqrtf(G * (h0));
    return CFL * fminf(DX, DY) / (c0 + 1e-6f);
}

void reduce_mass_min(const std::vector<float>& h_host, int pitch,
                     double& mass, float& hmin)
{
    mass = 0.0;
    hmin = 1e30f;
    for (int j = 1; j <= NY; ++j) {
        for (int i = 1; i <= NX; ++i) {
            float hh = h_host[IDX(i, j, pitch)];
            mass += hh;
            hmin = std::min(hmin, hh);
        }
    }
    mass *= (double)DX * (double)DY;
}

int main()
{
    // padded pitch = NX+2 (ghost ring)
    const int pitch = NX + 2;
    const int nTot  = (NX + 2) * (NY + 2);
    const size_t bytes = nTot * sizeof(float);

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

    // Launch geometry
    dim3 block(64,4);
    int nthreads_bc = 256;
    dim3 grid((NX + block.x - 1)/block.x, (NY + block.y - 1)/block.y);
    // compute_rhs_vec4x4 handles four cells in x per thread, so it needs a quarter of the grid in x.
    // The other kernels are still one cell per thread and must use the full grid.
    dim3 gridVec(((NX + 3)/4 + block.x - 1)/block.x, (NY + block.y - 1)/block.y);
    dim3 gridBC((std::max(NX, NY) + (nthreads_bc-1))/nthreads_bc);

    // Initialize fields: still water + Gaussian bump
    const float h0  = 1.0f;                  // base depth
    const float amp = 0.2f;                  // bump amplitude
    const float x0  = 0.5f * (NX-1) * DX;    // center
    const float y0  = 0.5f * (NY-1) * DY;
    const float sx  = 20.0f;                 // std dev in x
    const float sy  = 20.0f;                 // std dev in y

    init_gaussian<<<grid, block>>>(d_h, d_hu, d_hv, pitch, h0, amp, x0, y0, sx, sy);
    CHECK_CUDA(hipGetLastError());
    apply_reflect_bc<<<gridBC, nthreads_bc>>>(d_h, d_hu, d_hv, pitch);
    CHECK_CUDA(hipGetLastError());
    //CHECK_CUDA(hipDeviceSynchronize());

    // Host copy for accuracy baseline (mass)
    std::vector<float> h0_host(nTot);
    CHECK_CUDA(hipMemcpy(h0_host.data(), d_h, bytes, hipMemcpyDeviceToHost));
    double mass0; float hmin0;
    reduce_mass_min(h0_host, pitch, mass0, hmin0);

    // Time step
    float dt = initial_dt(h0 + amp); // conservative estimate from max initial depth
    // You can print dt if desired:
    // printf("dt = %.6g\n", dt);

    // Time stepping (measure performance)
    hipEvent_t e0, e1;
    CHECK_CUDA(hipEventCreate(&e0));
    CHECK_CUDA(hipEventCreate(&e1));
    CHECK_CUDA(hipDeviceSynchronize());
    CHECK_CUDA(hipEventRecord(e0));

    for (int n = 0; n < NSTEPS; ++n) {
        // k1
        apply_reflect_bc<<<gridBC, nthreads_bc>>>(d_h, d_hu, d_hv, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        compute_rhs_vec4x4<<<gridVec, block>>>(d_h, d_hu, d_hv, d_k1_h, d_k1_hu, d_k1_hv, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
	
        // k2
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                      d_h, d_hu, d_hv,
                                      d_k1_h, d_k1_hu, d_k1_hv, 0.5f*dt, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        apply_reflect_bc<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        compute_rhs_vec4x4<<<gridVec, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k2_h, d_k2_hu, d_k2_hv, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());

        // k3
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                      d_h, d_hu, d_hv,
                                      d_k2_h, d_k2_hu, d_k2_hv, 0.5f*dt, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        apply_reflect_bc<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        compute_rhs_vec4x4<<<gridVec, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k3_h, d_k3_hu, d_k3_hv, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());

        // k4
        update_stage<<<grid, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp,
                                      d_h, d_hu, d_hv,
                                      d_k3_h, d_k3_hu, d_k3_hv, 1.0f*dt, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        apply_reflect_bc<<<gridBC, nthreads_bc>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
        compute_rhs_vec4x4<<<gridVec, block>>>(d_h_tmp, d_hu_tmp, d_hv_tmp, d_k4_h, d_k4_hu, d_k4_hv, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
	
        // final update
        final_update<<<grid, block>>>(d_h, d_hu, d_hv,
                                      d_k1_h, d_k1_hu, d_k1_hv,
                                      d_k2_h, d_k2_hu, d_k2_hv,
                                      d_k3_h, d_k3_hu, d_k3_hv,
                                      d_k4_h, d_k4_hu, d_k4_hv,
                                      dt, pitch);
	//CHECK_CUDA(hipDeviceSynchronize());
    }

    CHECK_CUDA(hipEventRecord(e1));
    CHECK_CUDA(hipEventSynchronize(e1));
    float ms = 0.0f;
    CHECK_CUDA(hipEventElapsedTime(&ms, e0, e1));
    double seconds = 1e-3 * (double)ms;

    // Throughput (MCUPS): counts RK4 stages (4 rhs evaluations per step)
    double cells = (double)NX * (double)NY;
    double updates = cells * 4.0 * (double)NSTEPS;
    double mcups = updates / seconds / 1.0e6;

    // Accuracy check: mass conservation + positivity
    std::vector<float> h_host(nTot);
    CHECK_CUDA(hipMemcpy(h_host.data(), d_h, bytes, hipMemcpyDeviceToHost));
    double mass1; float hmin;
    reduce_mass_min(h_host, pitch, mass1, hmin);

    double rel_mass_err = fabs(mass1 - mass0) / fmax(1e-12, fabs(mass0));

    printf("Domain: %dx%d, steps=%d, dt=%.6g\n", NX, NY, NSTEPS, dt);
    printf("Elapsed: %.3f s  |  Throughput (including RK4 stages): %.2f MCUPS\n", seconds, mcups);
    printf("Mass: initial=%.9e, final=%.9e, rel.err=%.3e\n", mass0, mass1, rel_mass_err);
    printf("Min(h) after run: %.6g %s\n", hmin, (hmin < 0.f ? "(NEGATIVE!)" : ""));

    // Clean
    hipEventDestroy(e0); hipEventDestroy(e1);
    CHECK_CUDA(hipFree(d_h));
    CHECK_CUDA(hipFree(d_hu));
    CHECK_CUDA(hipFree(d_hv));
    CHECK_CUDA(hipFree(d_h_tmp));
    CHECK_CUDA(hipFree(d_hu_tmp));
    CHECK_CUDA(hipFree(d_hv_tmp));
    CHECK_CUDA(hipFree(d_k1_h));
    CHECK_CUDA(hipFree(d_k1_hu));
    CHECK_CUDA(hipFree(d_k1_hv));
    CHECK_CUDA(hipFree(d_k2_h));
    CHECK_CUDA(hipFree(d_k2_hu));
    CHECK_CUDA(hipFree(d_k2_hv));
    CHECK_CUDA(hipFree(d_k3_h));
    CHECK_CUDA(hipFree(d_k3_hu));
    CHECK_CUDA(hipFree(d_k3_hv));
    CHECK_CUDA(hipFree(d_k4_h));
    CHECK_CUDA(hipFree(d_k4_hu));
    CHECK_CUDA(hipFree(d_k4_hv));

    return 0;
}
