struct Var2D {
   int sizeY, sizeX;
   double *data;

   Var2D(int sizeY, int sizeX) : sizeY(sizeY), sizeX(sizeX) {
        data = (double *)aligned_alloc(128,sizeof(double) * sizeX * sizeY);
        //hipMallocManaged(&data,sizeof(double) * sizeX * sizeY);
   }

   double &operator()(int j, int i) const { return data[j*sizeX + i];}
   double *data_ptr() { return data; }
};
