// C version of reduction scalar reproducer created by Mahdieh Ghazimirsaeed
// Converted by Bob Robey
// Copyright (c) 2025 AMD HPC Application Performance Team
// MIT License

#include <stdio.h>
int main(int argc, char *argv[]){
   double ce1=0.0;
   double ce2=0.0;
   
   #pragma omp target teams distribute parallel for reduction(+:ce1,ce2)
   for (int j = 0; j< 1000; j++){
      ce1 += 1.0;
      ce2 += 1.0;
   }

   printf("ce1 = %lf ce2 %lf\n", ce1, ce2);
   return(0);
}
