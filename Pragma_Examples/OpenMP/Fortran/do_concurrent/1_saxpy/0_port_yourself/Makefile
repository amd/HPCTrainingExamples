EXEC = saxpy
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FREE_FORM_FLAG = -ffree-form
OMP_FLAG = -fopenmp
FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OMP_FLAG}
LDFLAGS=${FFLAGS}

${EXEC}: ${EXEC}.F08
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
