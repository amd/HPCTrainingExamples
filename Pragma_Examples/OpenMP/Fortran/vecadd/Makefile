EXEC = vecadd
default: ${EXEC}
all: ${EXEC}

FC1=$(notdir $(FC))

ifeq ($(findstring amdflang,$(FC1)), amdflang)
  OPENMP_FLAGS = -fopenmp --offload-arch=native
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring flang-new,$(FC1)), flang-new)
  OPENMP_FLAGS = -fopenmp --offload-arch=native
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=native
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  OPENMP_FLAGS = -fopenmp --offload=-march=native
  #OPENMP_FLAGS = -fopenmp -foffload=-march=native -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else
   LDFLAGS = ${OPENMP_FLAGS}
endif

ftimers.o: timers.o
${EXEC}.o: ftimers.o

${EXEC}: ${EXEC}.o timers.o ftimers.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
