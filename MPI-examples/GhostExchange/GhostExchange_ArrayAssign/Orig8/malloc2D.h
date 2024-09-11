#ifndef _MALLOC2D_H
#define _MALLOC2D_H
#ifdef __cplusplus
extern "C" {
#endif
   double **malloc2D(int jmax, int imax, int joffset, int ioffset);
   void malloc2D_free(double **x, int joffset);
#ifdef __cplusplus
}
#endif
#endif
