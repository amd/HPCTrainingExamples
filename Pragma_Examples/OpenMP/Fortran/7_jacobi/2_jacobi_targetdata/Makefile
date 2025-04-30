EXEC = jacobi
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifeq ($(findstring amdflang,$(FC1)), amdflang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -G2 -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifeq ($(findstring amdflang,$(FC1)), amdflang)
  FFLAGS = -g -O3 -ffast-math ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else ifeq ($(findstring ftn,$(FC1)), ftn)
   LDFLAGS = ${OPENMP_FLAGS} -L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12
else
   LDFLAGS = ${OPENMP_FLAGS}
endif

OBJS= \
  kind.o \
	mesh.o \
	input.o \
	norm.o \
	laplacian.o \
	boundary.o \
	update.o \
	jacobi.o \
	main.o \

boundary.o: mesh.o kind.o
input.o: mesh.o kind.o
jacobi.o: input.o mesh.o norm.o laplacian.o boundary.o update.o kind.o
laplacian.o: mesh.o kind.o
norm.o: mesh.o kind.o
update.o: mesh.o kind.o
main.o: mesh.o jacobi.o kind.o

%.o : %.f90 Makefile
	$(FC) $(FFLAGS) -c $< -o $@

${EXEC}: ${OBJS}
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
