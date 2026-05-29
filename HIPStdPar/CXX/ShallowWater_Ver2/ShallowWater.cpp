#define DEBUG 1
#include <chrono>
#include <execution>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "ShallowWater.h"

#include "Var2D.hpp"
/*********************************************************************************************
 * WAVE -- 2D Shallow Water Equation Model
 *                            Bob Robey, AMD
 * ******************************************************************************************/

//define macro for squaring a number
#define SQ(x) ((x)*(x))

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)

using namespace std::chrono;

int main(int argc, char *argv[])
{
  const int      nx = 6, ny = 4; //nx = 500, ny = 200;
  const int      ntimes = 2, nburst = 1; //ntimes = 2000, nburst = 100;
  const double   deltaX=1.0, deltaY=1.0;         //size of cell
  const double   g = 9.80;                       // gravitational constant
  const double   sigma = 0.95;
  double         time=0.0;                       //computer simulation time
  double         TotalMass, origTM;    //variables for checking conservation of mass
  double         *temp;

  /* allocate the memory dynamically for the matrix */
  // state variables
  Var2D H(ny+2, nx+2);
  Var2D U(ny+2, nx+2);
  Var2D V(ny+2, nx+2);

  Var2D Hnew(ny+2, nx+2);
  Var2D Unew(ny+2, nx+2);
  Var2D Vnew(ny+2, nx+2);

  // half-step arrays
  Var2D Hx(ny, nx+1);
  Var2D Ux(ny, nx+1);
  Var2D Vx(ny, nx+1);

  Var2D Hy(ny+1, nx);
  Var2D Uy(ny+1, nx);
  Var2D Vy(ny+1, nx);

  /*initialize matrix*/
  
  Range2D active_range(1,ny+1,1,nx+1); // defines 2D index space

  auto flatindexrange = range(0,ny*nx);
  std::for_each(
    std::execution::par_unseq,  // execution policy
    flatindexrange.begin(),     // iterator begin and end
    flatindexrange.end(),
    // begin functor or lambda
    [=](int flatindex) {
      const auto j = active_range.beginY + (flatindex / active_range.sizeX);
      const auto i = active_range.beginX + (flatindex % active_range.sizeX);

      if (i<= (nx+1)/2)
        H(j,i)=10.0 - ((10.0 - 2.0)/ (double)((nx+1)/2))*(double)(i);
      else
        H(j,i)=2.0;
      U(j,i)=0.0;
      V(j,i)=0.0;
    }
    // end functor or lambda
  );

#ifdef DEBUG
  printf("After initialization of data\n");
  for (int j=0; j<=ny+1; j++){
    for (int i=0; i<=nx+1; i++){
      printf(" i %d j %d H(j,i) %lf &H(j,i) - &H(0,0) %ld\n",i,j,H(j,i),&H(j,i) - &H(0,0));
    }
  }
#endif

  //calculate original total mass
  origTM = std::transform_reduce(
    std::execution::par_unseq,  // execution policy
    flatindexrange.begin(),     // iterator begin and end
    flatindexrange.end(),
    0.0,
    std::plus<>(),
    // begin functor or lambda
    [=](int flatindex) {
      const auto j = active_range.beginY + (flatindex / active_range.sizeX);
      const auto i = active_range.beginX + (flatindex % active_range.sizeX);
      double Mass = H(j,i);
      return Mass;
    }
  );

  //calculate timestep
  double deltaT = std::transform_reduce(
    std::execution::par_unseq,  // execution policy
    flatindexrange.begin(),     // iterator begin and end
    flatindexrange.end(),
    1.0e20,
    [](auto l, auto r){ return std::fmin(l,r); },
    // begin functor or lambda
    [=](int flatindex) {
      const auto j = active_range.beginY + (flatindex / active_range.sizeX);
      const auto i = active_range.beginX + (flatindex % active_range.sizeX);

      double wavespeed = sqrt(g*H(j,i));
      double xspeed = (fabs(U(j,i))+wavespeed)/deltaX;
      double yspeed = (fabs(V(j,i))+wavespeed)/deltaY;
      double my_deltaT = sigma/(xspeed+yspeed);
#ifdef DEBUG
      printf("i %d j %d H %lf dt %lf\n",i,j,H(j,i),my_deltaT);
#endif
      return my_deltaT;
    }
    // end functor or lambda
  );

  //print iteration info
  printf("Iteration:%5.5d, Time:%lf, Timestep:%lf Total mass:%lf\n", 0, time, deltaT, origTM);

  high_resolution_clock::time_point starttime = high_resolution_clock::now();

  /* run the simulation for given number of iterations */
  for (int n = 0; n < ntimes; ) {

    for (int ib=0; ib<nburst; ib++){

      //set boundary conditons
      auto bcyindexrange = range(0,ny+1);
      std::for_each(
        std::execution::par_unseq,  // execution policy
        bcyindexrange.begin(),     // iterator begin and end
        bcyindexrange.end(),
        // begin functor or lambda
        [=](int j) {
          H(j,0)=H(j,1);
          U(j,0)=-U(j,1);
          V(j,0)=V(j,1);
          H(j,nx+1)=H(j,nx);
          U(j,nx+1)=-U(j,nx);
          V(j,nx+1)=V(j,nx);
        }
        // end functor or lambda
      );

      auto bcxindexrange = range(0,nx+2);
      std::for_each(
        std::execution::par_unseq,  // execution policy
        bcxindexrange.begin(),     // iterator begin and end
        bcxindexrange.end(),
        // begin functor or lambda
        [=](int i) {
          H(0,i)=H(1,i);
          U(0,i)=U(1,i);
          V(0,i)=-V(1,i);
          H(ny+1,i)=H(ny,i);
          U(ny+1,i)=U(ny,i);
          V(ny+1,i)=-V(ny,i);
        }
        // end functor or lambda
      );

#ifdef DEBUG
  for (int j=0; j<=ny+1; j++){
    for (int i=0; i<=nx+1; i++){
      printf(" i %d j %d H(j,i) %lf &H(j,i) - &H(0,0) %ld\n",i,j,H(j,i),&H(j,i) - &H(0,0));
    }
  }
#endif

      //set timestep
      deltaT = std::transform_reduce(
        std::execution::par_unseq,  // execution policy
        flatindexrange.begin(),     // iterator begin and end
        flatindexrange.end(),
        1.0e20,
        [](auto l, auto r){ return std::fmin(l,r); },
        // begin functor or lambda
        [=](int flatindex) {
          const auto j = active_range.beginX + (flatindex / active_range.sizeX);
          const auto i = active_range.beginY + (flatindex % active_range.sizeX);

          double wavespeed = sqrt(g*H(j,i));
          double xspeed = (fabs(U(j,i))+wavespeed)/deltaX;
          double yspeed = (fabs(V(j,i))+wavespeed)/deltaY;
          double my_deltaT = sigma/(xspeed+yspeed);
#ifdef DEBUG
          printf("iter %d i %d j %d H %lf dt %lf\n",n,i,j,H(j,i),my_deltaT);
#endif

          return my_deltaT;
        }
         // end functor or lambda
      );

      //first pass
      //x direction
      Range2D xface_range(0,ny,0,nx+1); // defines 2D index space

      auto xfaceindexrange = range(0,ny*(nx+1));
      std::for_each(
        std::execution::par_unseq,  // execution policy
        xfaceindexrange.begin(),     // iterator begin and end
        xfaceindexrange.end(),
        // begin functor or lambda
        [=](int xfaceindex) {
          const auto j = xface_range.beginY + (xfaceindex / xface_range.sizeX);
          const auto i = xface_range.beginX + (xfaceindex % xface_range.sizeX);

          //density calculation
          Hx(j,i)=0.5*(H(j+1,i+1)+H(j+1,i  )) - deltaT/(2.0*deltaX)*
                      (U(j+1,i+1)-U(j+1,i  ));
          //momentum x calculation
          Ux(j,i)=0.5*(U(j+1,i+1)+U(j+1,i  )) - deltaT/(2.0*deltaX)*
                             ((SQ(U(j+1,i+1))/H(j+1,i+1) + 0.5*g*SQ(H(j+1,i+1))) -
                              (SQ(U(j+1,i  ))/H(j+1,i  ) + 0.5*g*SQ(H(j+1,i  ))));
          //momentum y calculation
          Vx(j,i)=0.5*(V(j+1,i+1)+V(j+1,i  )) - deltaT/(2.0*deltaX)*
                             ((U(j+1,i+1)*V(j+1,i+1)/H(j+1,i+1)) -
                              (U(j+1,i  )*V(j+1,i  )/H(j+1,i  )));
#ifdef DEBUG
          printf("iter %d i %d j %d Hx %lf Ux %lf Vx %lf\n",n,i,j,Hx(j,i),Ux(j,i),Vx(j,i));
#endif
        }
        // end functor or lambda
      );

      //y direction
      Range2D yface_range(0,ny+1,0,nx); // defines 2D index space

      auto yfaceindexrange = range(0,(ny+1)*nx);
      std::for_each(
        std::execution::par_unseq,  // execution policy
        yfaceindexrange.begin(),     // iterator begin and end
        yfaceindexrange.end(),
        // begin functor or lambda
        [=](int yfaceindex) {
          const auto j = yface_range.beginY + (yfaceindex / yface_range.sizeX);
          const auto i = yface_range.beginX + (yfaceindex % yface_range.sizeX);

          //density calculation
          Hy(j,i)=0.5*(H(j+1,i+1)+H(j  ,i+1)) - deltaT/(2.0*deltaY)*
                      (V(j+1,i+1)-V(j  ,i+1));
          //momentum x calculation
          Uy(j,i)=0.5*(U(j+1,i+1)+U(j  ,i+1)) - deltaT/(2.0*deltaY)*
                             ((V(j+1,i+1)*U(j+1,i+1)/H(j+1,i+1)) -
                              (V(j  ,i+1)*U(j  ,i+1)/H(j  ,i+1)));
          //momentum y calculation
          Vy(j,i)=0.5*(V(j+1,i+1)+V(j  ,i+1)) - deltaT/(2.0*deltaY)*
                             ((SQ(V(j+1,i+1))/H(j+1,i+1) + 0.5*g*SQ(H(j+1,i+1))) -
                              (SQ(V(j  ,i+1))/H(j  ,i+1) + 0.5*g*SQ(H(j  ,i+1))));
#ifdef DEBUG
          printf("iter %d i %d j %d Hy %lf Uy %lf Vy %lf\n",n,i,j,Hy(j,i),Uy(j,i),Vy(j,i));
#endif
        }
        // end functor or lambda
      );

      //second pass
      Range2D pass2_range(1,ny,1,nx); // defines 2D index space

      auto pass2indexrange = range(0,ny*nx);
      std::for_each(
        std::execution::par_unseq,  // execution policy
        flatindexrange.begin(),     // iterator begin and end
        flatindexrange.end(),
        // begin functor or lambda
        [=](int flatindex) {
          const auto j = active_range.beginY + (flatindex / active_range.sizeX);
          const auto i = active_range.beginX + (flatindex % active_range.sizeX);

          //density calculation
          Hnew(j,i) = H(j,i) - (deltaT/deltaX)*(Ux(j-1,i  )-Ux(j-1,i-1))
                             - (deltaT/deltaY)*(Vy(j  ,i-1)-Vy(j-1,i-1));
          //momentum x calculation
          Unew(j,i) = U(j,i) - (deltaT/deltaX)*
                                  ((SQ(Ux(j-1,i  ))/Hx(j-1,i  ) +0.5*g*SQ(Hx(j-1,i  ))) -
                                   (SQ(Ux(j-1,i-1))/Hx(j-1,i-1) +0.5*g*SQ(Hx(j-1,i-1))))
                             - (deltaT/deltaY)*
                                  ((Vy(j  ,i-1)*Uy(j  ,i-1)/Hy(j  ,i-1)) -
                                   (Vy(j-1,i-1)*Uy(j-1,i-1)/Hy(j-1,i-1)));
          //momentum y calculation
          Vnew(j,i) = V(j,i) - (deltaT/deltaX)*
                                  ((Ux(j-1,i  )*Vx(j-1,i  )/Hx(j-1,i  )) -
                                   (Ux(j-1,i-1)*Vx(j-1,i-1)/Hx(j-1,i-1)))
                             - (deltaT/deltaY)*
                                  ((SQ(Vy(j  ,i-1))/Hy(j  ,i-1) +0.5*g*SQ(Hy(j  ,i-1))) -
                                   (SQ(Vy(j-1,i-1))/Hy(j-1,i-1) +0.5*g*SQ(Hy(j-1,i-1))));
#ifdef DEBUG
          printf("iter %d i %d j %d Hnew %lf Unew %lf Vnew %lf\n",n,i,j,Hnew(j,i),Unew(j,i),Vnew(j,i));
#endif
        }
        // end functor or lambda
      );

      SWAP_PTR(H.data, Hnew.data, temp);
      SWAP_PTR(U.data, Unew.data, temp);
      SWAP_PTR(V.data, Vnew.data, temp);

#ifdef DEBUG
      for (int j=0; j<=ny+1; j++){
        for (int i=0; i<=nx+1; i++){
          printf(" i %d j %d H(j,i) %lf &H(j,i) - &H(0,0) %ld\n",i,j,H(j,i),&H(j,i) - &H(0,0));
        }
      }
#endif

    } // burst loop

    TotalMass = std::transform_reduce(
      std::execution::par_unseq,  // execution policy
      flatindexrange.begin(),     // iterator begin and end
      flatindexrange.end(),
      0.0,
      std::plus<>(),
      // begin functor or lambda
      [=](int flatindex) {
        const auto j = active_range.beginY + (flatindex / active_range.sizeX);
        const auto i = active_range.beginX + (flatindex % active_range.sizeX);
        double Mass = H(j,i);
        return Mass;
      }
    );

    if(((fabs(TotalMass-origTM)>1.0E-6)||std::isnan(TotalMass))&&check==1){
      printf("Conservation of mass\nMass difference:%e\n", TotalMass-origTM);
      printf("Problem occured on iteration %5.5d at time %f.\n", n, time);
      //exit(0);
    }
    time+=deltaT;
    n+=nburst;

    //print iteration info
    printf("Iteration:%5.5d, Time:%f, Timestep:%f Total mass:%f\n", n, time, deltaT, TotalMass);

  } // End of iteration loop

  /* Compute the average time taken/processor */
  high_resolution_clock::time_point endtime = high_resolution_clock::now();

  /* Print the total time taken */
  duration<double> totaltime = duration_cast<duration<double>>(endtime - starttime);
  std::cout << " Flow finished in " << totaltime.count() << " seconds" << std::endl;

  // Free memory allocated with malloc2D call
  free(H.data);
  free(U.data);
  free(V.data);
  free(Hnew.data);
  free(Unew.data);
  free(Vnew.data);
  free(Hx.data);
  free(Ux.data);
  free(Vx.data);
  free(Hy.data);
  free(Uy.data);
  free(Vy.data);
  
  exit(0);
}
