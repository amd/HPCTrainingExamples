from mpi4py import MPI
import cupy


def mpi4py_cupy_test():

   comm = MPI.COMM_WORLD
   size = comm.Get_size()
   rank = comm.Get_rank()

   # Allreduce
   if rank == 0:
     print("Starting allreduce test...")
   sendbuf = cupy.arange(10, dtype='i')
   recvbuf = cupy.empty_like(sendbuf)
   # always make sure the GPU buffer is ready before any MPI operation
   cupy.cuda.get_current_stream().synchronize()
   comm.Allreduce(sendbuf, recvbuf)
   assert cupy.allclose(recvbuf, sendbuf*size)

   # Bcast
   if rank == 0:
     print("Starting bcast test...")
   if rank == 0:
       buf = cupy.arange(100, dtype=cupy.complex64)
   else:
       buf = cupy.empty(100, dtype=cupy.complex64)
   cupy.cuda.get_current_stream().synchronize()
   comm.Bcast(buf)
   assert cupy.allclose(buf, cupy.arange(100, dtype=cupy.complex64))

   # Send-Recv
   if rank == 0:
     print("Starting send-recv test...")

   if rank == 0:
       buf = cupy.arange(20, dtype=cupy.float64)
       cupy.cuda.get_current_stream().synchronize()
       for j in range(1,size):
          comm.Send(buf, dest=j, tag=88+j)
   else:
       buf = cupy.empty(20, dtype=cupy.float64)
       cupy.cuda.get_current_stream().synchronize()
       comm.Recv(buf, source=0, tag=88+rank)
       assert cupy.allclose(buf, cupy.arange(20, dtype=cupy.float64))

   if rank == 0:
     print("Success")

#-------------------------------------------------------------------------------

if __name__ == "__main__":

    mpi4py_cupy_test()

