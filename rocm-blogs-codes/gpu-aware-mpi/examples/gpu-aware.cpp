#include <stdio.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <time.h>
#include <cstdlib>

int main(int argc, char **argv) {
	int i,rank,size,bufsize;
	int *h_buf;
	int *d_buf;
	MPI_Status status;

	bufsize=100;

        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

	//allocate buffers
	h_buf=(int*) malloc(sizeof(int)*bufsize);
	hipMalloc(&d_buf, bufsize*sizeof(int));

	//initialize buffers
	if(rank==0) {
	  for(i=0;i<bufsize;i++)
            h_buf[i]=i;
	}

	if(rank==1) {
	  for(i=0;i<bufsize;i++)
            h_buf[i]=-1;
	}

	hipMemcpy(d_buf, h_buf, bufsize*sizeof(int), hipMemcpyHostToDevice);

	//communication
	if(rank==0)
          MPI_Send(d_buf, bufsize, MPI_INT, 1, 123, MPI_COMM_WORLD);

	if(rank==1)
	  MPI_Recv(d_buf, bufsize, MPI_INT, 0, 123, MPI_COMM_WORLD, &status);

	//validate results
	if(rank==1) {
	  hipMemcpy(h_buf, d_buf, bufsize*sizeof(int), hipMemcpyDeviceToHost);
	  for(i=0;i<bufsize;i++) {
	    if(h_buf[i] != i)
	      printf("Error: buffer[%d]=%d but expected %d\n", i, h_buf[i], i);
	  }
	  fflush(stdout);
	}

	//free buffers
	free(h_buf);
	hipFree(d_buf);
	
	MPI_Finalize();
}
