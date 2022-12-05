EXEC = freduce
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

ifeq ($(notdir $(FC)), flang)
  OPENACC_FLAGS = Flang_does_not_support_OpenACC_for_AMD
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(notdir $(FC)), amdflang)
  OPENACC_FLAGS = amdflang_does_not_support_OpenACC_for_AMD
  FREE_FORM_FLAG = -Mfreeform
else
  OPENACC_FLAGS = -fopenacc -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
endif

FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENACC_FLAGS}
LDFLAGS = ${OPENACC_FLAGS}

${EXEC}: ${EXEC}.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
