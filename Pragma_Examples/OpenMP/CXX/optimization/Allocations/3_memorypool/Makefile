all: memorypool

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifneq ($(findstring amdclang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
else ifneq ($(findstring clang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
else ifneq ($(findstring g++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU}
else ifneq ($(findstring CXX,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp
endif

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS} -D__HIP_PLATFORM_AMD__ -I${UMPIRE_PATH}/include
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm -L${UMPIRE_PATH}/lib64 -L${UMPIRE_PATH}/lib -lumpire -lcamp -lfmt -L${ROCM_PATH}/lib -lamdhip64 -lstdc++

memorypool: memorypool.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o memorypool
	rm -rf build
