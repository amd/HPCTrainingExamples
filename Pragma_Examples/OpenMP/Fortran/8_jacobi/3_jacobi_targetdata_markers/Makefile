EXEC = jacobi
default: ${EXEC}
all: ${EXEC}

ROCM_PATH ?= /opt/rocm
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  OPENMP_FLAGS = -fopenmp
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    FREE_FORM_FLAG = -f free
    FFLAGS = -eZ
  else
    FREE_FORM_FLAG = -ffree-form
    FFLAGS =
  endif
endif

FFLAGS := -g -O3 ${FFLAGS} ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib

OBJS= \
  kind.o \
	mesh.o \
	input.o \
	norm.o \
	laplacian.o \
	boundary.o \
	update.o \
	roctx_mod.o \
	jacobi.o \
	main.o \

boundary.o: mesh.o kind.o
input.o: mesh.o kind.o
jacobi.o: input.o mesh.o norm.o laplacian.o boundary.o update.o roctx_mod.o kind.o
laplacian.o: mesh.o kind.o
norm.o: mesh.o kind.o
roctx_mod.o:
update.o: mesh.o kind.o
main.o: mesh.o jacobi.o kind.o

%.o : %.f90 Makefile
	$(FC) $(FFLAGS) -c $< -o $@

${EXEC}: ${OBJS}
	$(FC) $(LDFLAGS) $^ -o $@ -lrocprofiler-sdk-roctx

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
