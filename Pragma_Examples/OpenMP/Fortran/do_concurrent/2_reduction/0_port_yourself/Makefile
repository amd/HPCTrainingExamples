EXEC = freduce
default: ${EXEC}
all: ${EXEC}

FREE_FORM_FLAG = -ffree-form
FFLAGS = -g -O3 ${FREE_FORM_FLAG}

${EXEC}: ${EXEC}.F08
	$(FC) $(FFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
