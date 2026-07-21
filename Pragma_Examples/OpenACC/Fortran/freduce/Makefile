EXEC = freduce
default: ${EXEC}
all: ${EXEC}

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang, $(FC1)),)
  OPENACC_FLAGS = -fopenacc --offload-arch=native
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring flang, $(FC1)),)
  OPENACC_FLAGS = -fopenacc --offload-arch=native
  FREE_FORM_FLAG = -Mfreeform
else
  OPENACC_FLAGS = -fopenacc -foffload=-march=native -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
endif

FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENACC_FLAGS}
ifeq (${FC1},gfortran-13)
  LDFLAGS = ${OPENACC_FLAGS} -fno-lto
else
  LDFLAGS = ${OPENACC_FLAGS}
endif

${EXEC}: ${EXEC}.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
