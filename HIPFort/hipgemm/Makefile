all: gemm_global gemm_local gemm_global_sd gemm_local_sd

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFORT_PATH ?= ${ROCM_PATH}

gemm_global:
	amdflang -g -c -fopenmp --offload-arch=${ROCM_GPU} gemm_mod.f90 -I${HIPFORT_PATH}/include/hipfort/amdgcn
	amdflang -g -fopenmp --offload-arch=${ROCM_GPU} -o gemm_global gemm_prog.f90 gemm_mod.o -I${HIPFORT_PATH}/include/hipfort/amdgcn -L${HIPFORT_PATH}/lib -lhipfort-amdgcn -L${ROCM_PATH}/lib -lhipblas -lamdhip64

gemm_local:
	amdflang -DLOCAL_ALLOC -g -c -fopenmp --offload-arch=${ROCM_GPU} gemm_mod.f90 -I${HIPFORT_PATH}/include/hipfort/amdgcn
	amdflang -DLOCAL_ALLOC -g -fopenmp --offload-arch=${ROCM_GPU} -o gemm_local gemm_prog.f90 gemm_mod.o -I${HIPFORT_PATH}/include/hipfort/amdgcn -L${HIPFORT_PATH}/lib -lhipfort-amdgcn -L${ROCM_PATH}/lib -lhipblas -lamdhip64

gemm_global_sd:
	amdflang -DSINGLE_DIRECTIVE -g -c -fopenmp --offload-arch=${ROCM_GPU} gemm_mod.f90 -I${HIPFORT_PATH}/include/hipfort/amdgcn
	amdflang -DSINGLE_DIRECTIVE -g -fopenmp --offload-arch=${ROCM_GPU} -o gemm_global_sd gemm_prog.f90 gemm_mod.o -I${HIPFORT_PATH}/include/hipfort/amdgcn -L${HIPFORT_PATH}/lib -lhipfort-amdgcn -L${ROCM_PATH}/lib -lhipblas -lamdhip64
gemm_local_sd:
	amdflang -DSINGLE_DIRECTIVE -DLOCAL_ALLOC -g -c -fopenmp --offload-arch=${ROCM_GPU} gemm_mod.f90 -I${HIPFORT_PATH}/include/hipfort/amdgcn
	amdflang -DSINGLE_DIRECTIVE -DLOCAL_ALLOC -g -fopenmp --offload-arch=${ROCM_GPU} -o gemm_local_sd gemm_prog.f90 gemm_mod.o -I${HIPFORT_PATH}/include/hipfort/amdgcn -L${HIPFORT_PATH}/lib -lhipfort-amdgcn -L${ROCM_PATH}/lib -lhipblas -lamdhip64

clean:
	rm -rf *.o rm -rf *.mod gemm_global gemm_local gemm_global_sd gemm_local_sd
