# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

add_one: add_one.o
	${HIPCC} ${HIPFLAGS} add_one.o -o add_one

add_one.o: add_one.cpp
	${HIPCC} ${HIPFLAGS} -c add_one.cpp

.PHONY: clean

clean:
	rm -f add_one *.o
