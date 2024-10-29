EXEC = interop
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

# Compiler
HIPCC				= clang -x hip
FC					= amdflang-new

# Compiler flags
HIP_FLAGS		= -O3 --offload-arch=${ROCM_GPU}
F_FLAGS			= -O3 -fopenmp --offload-arch=${ROCM_GPU}
LD_FLAGS		= -O3 -L${ROCM_PATH}/lib -lamdhip64 -lstdc++


${EXEC}:
	mkdir -p obj
	${HIPCC} ${HIP_FLAGS} -c kernel.cpp -o obj/kernel.o
	${FC} ${F_FLAGS} -c driver.f90 -o obj/driver.o
	${FC} ${F_FLAGS} ${LD_FLAGS} -o ${EXEC} obj/driver.o obj/kernel.o
	rm -rf obj
clean:
	rm -rf ${EXEC} obj *.mod *.o
