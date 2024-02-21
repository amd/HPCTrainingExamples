all: gemm_global gemm_local gemm_global_sd gemm_local_sd

gemm_global:
	amdflang -g -c -fopenmp --offload-arch=gfx90a gemm_mod.f90 -I${ROCM_PATH}/include/hipfort/amdgcn
	amdflang -g -fopenmp --offload-arch=gfx90a -o gemm_global gemm_prog.f90 gemm_mod.o -I${ROCM_PATH}/include/hipfort/amdgcn -L${ROCM_PATH}/lib -lhipfort-amdgcn -lhipblas -lamdhip64

gemm_local:
	amdflang -DLOCAL_ALLOC -g -c -fopenmp --offload-arch=gfx90a gemm_mod.f90 -I${ROCM_PATH}/include/hipfort/amdgcn
	amdflang -DLOCAL_ALLOC -g -fopenmp --offload-arch=gfx90a -o gemm_local gemm_prog.f90 gemm_mod.o -I${ROCM_PATH}/include/hipfort/amdgcn -L${ROCM_PATH}/lib -lhipfort-amdgcn -lhipblas -lamdhip64

gemm_global_sd:
	amdflang -DSINGLE_DIRECTIVE -g -c -fopenmp --offload-arch=gfx90a gemm_mod.f90 -I${ROCM_PATH}/include/hipfort/amdgcn
	amdflang -DSINGLE_DIRECTIVE -g -fopenmp --offload-arch=gfx90a -o gemm_global_sd gemm_prog.f90 gemm_mod.o -I${ROCM_PATH}/include/hipfort/amdgcn -L${ROCM_PATH}/lib -lhipfort-amdgcn -lhipblas -lamdhip64
gemm_local_sd:
	amdflang -DSINGLE_DIRECTIVE -DLOCAL_ALLOC -g -c -fopenmp --offload-arch=gfx90a gemm_mod.f90 -I${ROCM_PATH}/include/hipfort/amdgcn
	amdflang -DSINGLE_DIRECTIVE -DLOCAL_ALLOC -g -fopenmp --offload-arch=gfx90a -o gemm_local_sd gemm_prog.f90 gemm_mod.o -I${ROCM_PATH}/include/hipfort/amdgcn -L${ROCM_PATH}/lib -lhipfort-amdgcn -lhipblas -lamdhip64

clean:
	rm -rf *.o rm -rf *.mod gemm_global gemm_local gemm_global_sd gemm_local_sd
