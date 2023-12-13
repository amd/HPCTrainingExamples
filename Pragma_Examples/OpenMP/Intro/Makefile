# saxpy6 crashes during execution
all: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy2f saxpy8 targetdata

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifeq ($(findstring amdclang++,$(CXX1)), amdclang++)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifeq ($(findstring clang++,$(CXX1)), clang++)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifeq ($(findstring g++,$(CXX1)), g++)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifeq ($(findstring CC,$(CXX1)), CC)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}

ifeq ($(findstring amdflang,$(FC1)), amdflang)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring flang-new,$(FC1)), flang-new)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  FREE_FORM_FLAG = -ffree-form
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -g -O3 ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
endif

ifeq (${CXX1},g++-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

# From slide 3 -- CPU version
saxpy_cpu.o: saxpy_cpu.cpp
	$(CXX) -c -g -O3 -fstrict-aliasing saxpy_cpu.cpp -o saxpy_cpu.o

saxpy_cpu: saxpy_cpu.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 8 but without map clause
saxpy1: saxpy1.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 8 with map clause
saxpy2: saxpy2.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 9, Fortran example
# .f90 or .F90 extension sets code to free form
# some additional code has been added to time the loop and a test
# is added to avoid compilers from optimizing out the operation
# Capitalization of file extension, .F90, turns on preprocessing
%.o: %.F90
	$(FC) -c $(FFLAGS) $(FREE_FORM_FLAG) $< -o $@

saxpy1f: saxpy1f.o
	$(FC) $(LDFLAGS) $^ -o $@

# From slide 10 with optimized map clause
saxpy3: saxpy3.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 11 using subroutine call with pointers
saxpy4: saxpy4.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 14 adding parallel for simd
# simd is not recognized by AMD and Nvidia, but others such as Intel still use it
# parallel construct is a triplet of "parallel for simd"
saxpy5: saxpy5.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 16 adding parallel for simd
saxpy6: saxpy6.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 17
# adding teams distribute to hardware levels
# hardware levels is a triplet of "target teams distribute"
saxpy7: saxpy7.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 17 -- Fortran example
saxpy2f: saxpy2f.o
	$(FC) $(LDFLAGS) $^ -o $@

# From slide 19
saxpy8: saxpy8.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 22
targetdata: targetdata.o
	$(CXX) $(LDFLAGS) $^ -o $@

run: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy8 targetdata
	./saxpy_cpu
	./saxpy1
	./saxpy2
	./saxpy1f
	./saxpy3
	./saxpy4
	./saxpy5
	#./saxpy6
	./saxpy7
	./saxpy8
	./targetdata

clean:
	rm -f saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy5a saxpy6 saxpy7 saxpy2f saxpy8 targetdata *.o
