// C version of reduction scalar reproducer created by Mahdieh Ghazimirsaeed
// Converted by Bob Robey
// Copyright (c) 2024 AMD HPC Application Performance Team
// MIT License

#include <stdio.h>

int main(int argc, char *argv[]){
   double ce[2]={0.0, 0.0};
   for (int j = 0; j< 1000; j++){
      ce[0] += 1.0;
      ce[1] += 1.0;
   }

   printf("ce[0] = %lf ce[1] = %lf\n", ce[0], ce[1]);
   return(0);
}
