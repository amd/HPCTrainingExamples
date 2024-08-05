#pragma once

__global__ void initialize_polynomial_kernel(float *out, float3 a, float3 s, int x0, int x1, int y0, int y1, int z0, int z1,
        int line, int slice) {

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < x0 || i >= x1 || j < y0 || j >= y1 || k < z0 || k >= z1) return;

    size_t pos = POS(i,j,k);

    out[pos] = a.x * pow(i, s.x) + a.y * pow(j, s.y) + a.z * pow(k, s.z);

}


void initialize_polynomial(float *out, float3 a, float3 s, int x0, int x1, int y0, int y1, int z0, int z1,
        int line, int slice) {

        dim3 block (256);
        dim3 grid (ceil(x1, block.x), ceil(y1, block.y), ceil(z1, block.z));

        initialize_polynomial_kernel<<<grid, block>>>(out, a, s, x0, x1, y0, y1, z0, z1, line, slice);
        HIP_CHECK(hipGetLastError());

}
