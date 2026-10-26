EXEC = workshare_cpu
default: ${EXEC}
all: ${EXEC}

ifeq ($(origin FC), default)
  FC = amdflang
endif

ROCM_GPU ?= gfx942

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -ffree-form
else
  $(error This OpenMP 6.0 workdistribute example requires amdflang/flang with -fopenmp-version=60)
endif

FFLAGS := -g -O3 ${FFLAGS} ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

${EXEC}: ${EXEC}.f90
	$(FC) $(FFLAGS) $^ $(LDFLAGS) -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
