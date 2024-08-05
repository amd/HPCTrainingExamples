#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <assert.h>
#include <hip/hip_runtime.h>

#ifndef RADIUS
#define RADIUS 4
#endif

#include "helper.hpp"
#include "compare.hpp"
#include "compute_fd_x.hpp"
#include "compute_fd_x_vec.hpp"
#include "compute_fd_y.hpp"
#include "compute_fd_y_vec.hpp"
#include "compute_fd_y_lds_vec.hpp"
#include "compute_fd_xy_lds.hpp"
#include "compute_fd_xy_lds_vec.hpp"
#include "compute_fd_z.hpp"
#include "compute_fd_z_window.hpp"
#include "compute_fd_z_window_vec.hpp"
#include "compute_fd_xyz_lds_window.hpp"
#include "compute_fd_xyz_lds_window_vec.hpp"
#include "fd_coefficients.hpp"
#include "initialize.hpp"

int main(int argc, char **argv) {

    // Default grid size and number of iterations
    int nx = 512;
    int ny = 512;
    int nz = 512;
    int nt = 100;
    int nw = 1;
    // Default alignment factor that the leading dimension needs to be a multiple of.
    int align = 1;

    // Parse command line arguments (if provided)
    if (argc > 1) 
        nx = atof(argv[1]);
    if (argc > 2) 
        ny = atof(argv[2]);
    if (argc > 3) 
        nz = atof(argv[3]);
    if (argc > 4) 
        nt = atof(argv[4]);
    if (argc > 5) 
        nw = atof(argv[5]);
    if (argc > 6) 
        align = atoi(argv[6]);
    
    printf("Settings: nx = %d ny = %d nz = %d nt = %d nw = %d align = %d\n",
            nx, ny, nz, nt, nw, align);

    // Finite difference coefficients
    const int R = RADIUS;
    float h = 1.0;
    float d[2 * R + 1];

    // Finite difference approximation to use
    switch (R) {
        case 1:
            // 2nd order
            fd_coefficients_d2_order_2(d, R, h);
            break;
        case 2:
            // 4th order
            fd_coefficients_d2_order_4(d, R, h);
            break;
        case 3:
            // 6th order
            fd_coefficients_d2_order_6(d, R, h);
            break;
        case 4:
            // 8th order
            fd_coefficients_d2_order_8(d, R, h);
            break;
        default:
            printf("ERROR: unsupported radius (-DRADIUS=%d), exiting program...\n", R);
            exit(0);
    }

    // Padded grid size
    int mx = nx + 2 * XWIN;
    // Make the leading dimension a multiple of align
    // Align is typically chosen as a multiple of the cache line size (64 B)
    mx = ceil(mx, align) * align;
    const int my = ny + 2 * R;
    const int mz = nz + 2 * R;

    const int line = mx;
    const int slice = mx * my;

    // Total number of grid points
    const size_t m = (size_t)mx * (size_t)my * (size_t)mz;

    // Align the offset, if any
    int offset = align ? align - R : 0;
    offset = offset < 0 ? 0 : offset;

    float *d_p_in; alloc(&d_p_in, m, offset);
    float *d_p_out; alloc(&d_p_out, m, offset);
    float *d_p_ref; alloc(&d_p_ref, m, offset);
    
    // Initialize input, output, and reference arrays
    // initialize a polynomial of the form: a.x * x^s.x + a.y * y^s.y + a.z * z^s.z
    // i.e., x + y + z
    float3 a = {1.0f, 1.0f, 1.0f};
    float3 s = {1.0f, 1.0f, 1.0f};
    initialize_polynomial(d_p_in, a, s, 0, mx, 0, my, 0, mz, line, slice);
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    HIP_CHECK(hipMemset(d_p_ref, 0, m * sizeof(float)));
    //print_array(d_p_in, line, slice, 0, nx + 2 * R, 0, ny +2 * R, R, R + 1); 

    HIP_CHECK(hipDeviceSynchronize());

    init_fd_xy_gpu<R>(d, d);
    init_fd_z_gpu<R>(d);
    
    // Performance metrics
    float total_elapsed, elapsed, maxval;
    hipEvent_t start, stop;
    double theoretical_fetch_x = nx * ny * nz + 2 * R * ny * nz;
    double theoretical_fetch_y = nx * ny * nz + 2 * R * nx * nz;
    double theoretical_fetch_z = nx * ny * nz + 2 * R * nx * ny;
    double theoretical_fetch_xy = nx * ny * nz + 2 * R * nz * (ny + nx);
    double theoretical_fetch_xyz = nx * ny * nz + 2 * R * (ny * nz + nx * nz + nx * ny);
    double theoretical_write = nx * ny * nz;
    double total_bytes_x = (theoretical_fetch_x + theoretical_write) * sizeof(float); // GB 
    double total_bytes_y = (theoretical_fetch_y + theoretical_write) * sizeof(float); // GB 
    double total_bytes_z = (theoretical_fetch_z + theoretical_write) * sizeof(float); // GB 
    double total_bytes_xy = (theoretical_fetch_xy + theoretical_write) * sizeof(float); // GB 
    double total_bytes_xyz = (theoretical_fetch_xyz + theoretical_write) * sizeof(float); // GB 
    HIP_CHECK( hipEventCreate(&start) );
    HIP_CHECK( hipEventCreate(&stop)  );
    
    printf("\nApplying stencil in the x-direction: baseline implementation... \n");
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_x<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_x * nt / total_elapsed / 1e6);
    
    if (VEC_LEN > 1) {
    printf("\nApplying stencil in the x-direction: vectorized... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_x_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_y * nt / total_elapsed / 1e6);
    }
    }
    
    printf("\nApplying stencil in the y-direction: baseline implementation... \n");
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_y<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_y * nt / total_elapsed / 1e6);
    
    if (VEC_LEN > 1) {
    printf("\nApplying stencil in the y-direction: vectorized... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_y_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_y * nt / total_elapsed / 1e6);
    }
    }
    
    printf("\nApplying stencil in the y-direction: vectorized lds... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_y_lds_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_y * nt / total_elapsed / 1e6);
    }
    
    printf("\nApplying stencil in the z-direction: baseline implementation... \n");
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_z<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_z * nt / total_elapsed / 1e6);
    
    printf("\nApplying stencil in the z-direction: - sliding window... \n");
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_z_window<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz, nw);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_z * nt / total_elapsed / 1e6);
    
    
    if (VEC_LEN > 1) {
    printf("\nApplying stencil in the z-direction: - vectorized sliding window... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_z_window_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz, nw);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_z * nt / total_elapsed / 1e6);
    }
    }
    
    printf("\nApplying stencil in the xy-direction: vectorized lds... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_xy_lds_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_xy * nt / total_elapsed / 1e6);
    }
    
    //if (VEC_LEN > 1) {
    printf("\nApplying stencil in the xyz-direction: - vectorized lds & sliding window... \n");
    if (nx % VEC_LEN) printf("\nWarning: nx = %d not divisable by vec length %d, skipping\n",nx,VEC_LEN);
    else {
    HIP_CHECK(hipMemset(d_p_out, 0, m * sizeof(float)));
    total_elapsed = 0;
    for (int i = 0; i < nt; ++i) {
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        compute_fd_xyz_lds_window_vec<R>(d_p_out, d_p_in, d, line, slice, R, nx + R, R, ny + R, R, R + nz, nw);
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }
    maxval = max_absval_gpu(d_p_out, d_p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("\tMaximum absolute pointwise difference: %g \n", maxval);
    printf("\tAverage kernel time: %g ms\n", total_elapsed / nt);
    printf("\tEffective memory bandwidth %g GB/s \n",total_bytes_xyz * nt / total_elapsed / 1e6);
    }
    //}

    dealloc(d_p_in, offset);
    dealloc(d_p_out, offset);
    dealloc(d_p_ref, offset);

}
