ROCMVER=5.7.0
BASIC  	= -O0
OPT    	= -O3 -mtune=native -march=native
OMP 	= -fopenmp
CFLAGS = ${OPT} -g

all: rzf.x rzf_specialized.x rzf_function.x rzf_openmp.x rzf_mpi.x \
			rzf_fortran.x rzf_avx2.x rzf_ROCM.x rzf_ROCM_shard.x \
			rzf_fortran_openmp.x


rzf.x: rzf.c
	gcc ${CFLAGS} $< -o rzf.x -lm

rzf_avx2.x: rzf_avx2.c
	gcc ${CFLAGS} $< -o rzf_avx2.x -lm

rzf_specialized.x: rzf_specialized.c
	gcc ${CFLAGS} $< -o rzf_specialized.x -lm

rzf_function.x: rzf_function.c
	gcc ${CFLAGS} $< -o rzf_function.x -lm

rzf_openmp.x: rzf_openmp.c
	gcc ${CFLAGS} ${OMP} $< -o rzf_openmp.x -lm

rzf_mpi.x: rzf_mpi.c
	mpicc ${CFLAGS} $< -o rzf_mpi.x -lm

rzf_fortran.x: rzf.f
	gfortran $(CFLAGS) -ffree-form $< -o rzf_fortran.x

rzf_fortran_openmp.x: rzf_openmp.f
	gfortran $(CFLAGS) -fopenmp -ffree-form $< -o rzf_fortran_openmp.x



rzf_ROCM.x: rzf_ROCM.hip
	hipcc -O3 -march=native -mtune=native -I/opt/rocm-${ROCMVER}/rocprim rzf_ROCM.hip -o rzf_ROCM.x -lm

rzf_ROCM_shard.x: rzf_ROCM_shard.hip
	hipcc -O3 -march=native -mtune=native -I/opt/rocm-${ROCMVER}/rocprim rzf_ROCM_shard.hip -o rzf_ROCM_shard.x -lm




clean:
	rm -f *.x
