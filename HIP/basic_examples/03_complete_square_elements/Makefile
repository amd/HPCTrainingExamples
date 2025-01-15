# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

square_elements: square_elements.o
	${HIPCC} ${HIPFLAGS} square_elements.o -o square_elements

square_elements.o: square_elements.cpp
	${HIPCC} ${HIPFLAGS} -c square_elements.cpp

.PHONY: clean

clean:
	rm -f square_elements *.o
