#include "hip/hip_runtime.h"

__global__ void kernel (double *phi, double *laplacian_phi,
			double *grad_phi_x, double *grad_phi_y, double *grad_phi_z,
			double *f0, double *f1, double *f2, double *f3, double *f4,
			double *f5, double *f6,
			double *g0, double *g1, double *g2, double *g3, double *g4,
			double *g5, double *g6, double* g7, double *g8, double *g9,
			double *g10, double *g11, double *g12, double *g13, double *g14,
			double *g15, double *g16, double *g17, double *g18,
			int nx, int ny, int nz, int ldx, int ldy, int current, int next,
			double k, double alpha, double phi2, double gamma,
			double itauphi, double itauphi1, double ieta,
			double itaurho, double grav,
			double eg1, double eg2, double eg0, double egc0, double egc1, double egc2)
{
  int i = (threadIdx.x + blockIdx.x * blockDim.x);
  int j = (threadIdx.y + blockIdx.y * blockDim.y);
  int z = (threadIdx.z + blockIdx.z * blockDim.z);
    
  int m, current_pos;

  double mu_phi, current_phi, current_phi_2;
  double rho;
  double fx, fy, fz;
  double uf, ux, uy, uz, v;
  double af, ag, cf;
  double eg1ag, eg2ag, eg1rho, eg2rho;
  double tmp1, tmp2;

  if(i <= nx && j <= ny && z <= nz)
    {
      m = i + ldx * (j + ldy * z);
      current_pos = m + current;
      
      current_phi = phi[m];
      current_phi_2 = current_phi * current_phi;
      
      rho = g0[m] + g1[current_pos] + g2[current_pos] + g3[current_pos] + g4[current_pos] +
	g5[current_pos] + g6[current_pos] + g7[current_pos] + g8[current_pos]  + g9[current_pos] +
	g10[current_pos] + g11[current_pos] + g12[current_pos] + g13[current_pos] + g14[current_pos] +
	g15[current_pos] + g16[current_pos] + g17[current_pos] + g18[current_pos];
      
      mu_phi = alpha * current_phi * ( current_phi_2 - phi2 ) - k * laplacian_phi[m];

      fx = mu_phi * grad_phi_x[m];
      fy = mu_phi * grad_phi_y[m];
      fz = mu_phi * grad_phi_z[m];

      ux = ( g1[current_pos] - g2[current_pos] + g7[current_pos] - g8[current_pos] + g9[current_pos] -
	     g10[current_pos] + g11[current_pos] - g12[current_pos] + g13[current_pos] - g14[current_pos] +
	     0.50 * fx ) * 1.0/rho;
      uy = ( g3[current_pos] - g4[current_pos] + g7[current_pos] - g8[current_pos] - g9[current_pos] +
	     g10[current_pos] + g15[current_pos] - g16[current_pos] + g17[current_pos] - g18[current_pos] +
	     0.50 * fy ) * 1.0/rho;
      uz = ( g5[current_pos] - g6[current_pos] + g11[current_pos] - g12[current_pos] - g13[current_pos] +
	     g14[current_pos] + g15[current_pos] - g16[current_pos] - g17[current_pos] + g18[current_pos] +
	     0.50 * fz ) * 1.0/rho;
      
      af = 0.50 * gamma * mu_phi * itauphi;
      cf = itauphi * ieta * current_phi;
      
      f0[m] = itauphi1 * f0[m] + -3.0 * gamma * mu_phi * itauphi + itauphi * current_phi;
      
      f1[current_pos] = itauphi1 * f1[current_pos] + af + cf * ux;
      f2[current_pos] = itauphi1 * f2[current_pos] + af - cf * ux;
      f3[current_pos] = itauphi1 * f3[current_pos] + af + cf * uy;
      f4[current_pos] = itauphi1 * f4[current_pos] + af - cf * uy;
      f5[current_pos] = itauphi1 * f5[current_pos] + af + cf * uz;
      f6[current_pos] = itauphi1 * f6[current_pos] + af - cf * uz;

      ag  = 3.0 * current_phi * mu_phi + rho;
      eg1ag = eg1 * ag;
      eg2ag = eg2 * ag;
      eg1rho = eg1 * rho;
      eg2rho = eg2 * rho;
      v  = 1.50 * ( ux*ux + uy*uy + uz*uz );
      uf = ux * fx + uy * fy + uz * fz;

      g0[m] = itaurho * g0[m] + eg0 * ( (rho - 6.0 * current_phi * mu_phi) - rho*v ) - egc0*uf;
      
      tmp1 = eg1ag + eg1rho*( 0.50*ux*ux - v ) + egc1*( ux*fx - uf );
      tmp2 = eg1rho*ux + egc1*fx;
      
      g1[m+next + 1] = itaurho * g1[current_pos] + tmp1 + tmp2;
      g2[m+next - 1] = itaurho * g2[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg1ag + eg1rho*( 0.50 * uy * uy - v ) + egc1 * ( uy * fy - uf );
      tmp2 = eg1rho * uy + egc1 * fy;
      
      g3[m+next + ldx] = itaurho * g3[current_pos] + tmp1 + tmp2;
      g4[m+next - ldx] = itaurho * g4[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg1ag + eg1rho*( 0.50 * uz * uz - v ) + egc1 * ( uz * fz - uf );
      tmp2 = eg1rho * uz + egc1 * fz;
      
      g5[m+next + ldx*ldy] = itaurho * g5[current_pos] + tmp1 + tmp2;
      g6[m+next - ldx*ldy] = itaurho * g6[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg2ag + eg2rho * ( 0.50 * ( ux + uy ) * ( ux + uy ) - v ) +
	egc2 * ( ( ux + uy ) * ( fx + fy ) - uf );
      
      tmp2 = eg2rho * ( ux + uy ) + egc2 * ( fx + fy );
      
      g7[m+next + 1 + ldx] = itaurho * g7[current_pos] + tmp1 + tmp2;
      g8[m+next - 1 - ldx] = itaurho * g8[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg2ag + eg2rho * ( 0.50 * ( ux - uy ) * ( ux - uy ) - v ) +
	egc2 * ( ( ux - uy )*( fx - fy ) - uf );
      tmp2 = eg2rho * ( ux - uy ) + egc2 * ( fx - fy );
      
      g9[m+next + 1 - ldx]  = itaurho * g9[current_pos]  + tmp1 + tmp2;
      g10[m+next - 1 + ldx] = itaurho * g10[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg2ag + eg2rho * ( 0.50 * ( ux + uz ) * ( ux + uz ) - v ) +
	egc2 * ( ( ux + uz ) * ( fx + fz ) - uf );
      tmp2 = eg2rho * ( ux + uz ) + egc2 * ( fx + fz );
      
      g11[m+next + 1 + ldx*ldy] = itaurho * g11[current_pos] + tmp1 + tmp2;
      g12[m+next - 1 - ldx*ldy] = itaurho * g12[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg2ag + eg2rho * ( 0.50 * ( ux - uz ) * ( ux - uz ) - v ) +
	egc2 * ( ( ux - uz ) * ( fx - fz ) - uf );
      tmp2 = eg2rho * ( ux - uz ) + egc2 * ( fx - fz );
      
      g13[m+next + 1 - ldx*ldy] = itaurho * g13[current_pos] + tmp1 + tmp2;
      g14[m+next - 1 + ldx*ldy] = itaurho * g14[current_pos] + tmp1 - tmp2;
      
      tmp1 = eg2ag + eg2rho * ( 0.50 * ( uy + uz ) * ( uy + uz ) - v ) +
	egc2 * ( ( uy + uz ) * ( fy + fz ) - uf );
      tmp2 = eg2rho * ( uy + uz ) + egc2 * ( fy + fz );
      
      g15[m+next + ldx + ldx*ldy] = itaurho * g15[current_pos] + tmp1 + tmp2;
      g16[m+next - ldx - ldx*ldy] = itaurho * g16[current_pos] + tmp1 - tmp2;

      tmp1 = eg2ag + eg2rho * ( 0.50 * ( uy - uz ) * ( uy - uz ) - v ) +
	egc2 * ( ( uy - uz ) * ( fy - fz ) - uf );
      tmp2 = eg2rho * ( uy - uz ) + egc2 * ( fy - fz );
      
      g17[m+next + ldx - ldx*ldy] = itaurho * g17[current_pos] + tmp1 + tmp2;
      g18[m+next - ldx + ldx*ldy] = itaurho * g18[current_pos] + tmp1 - tmp2;
      
    }
}
