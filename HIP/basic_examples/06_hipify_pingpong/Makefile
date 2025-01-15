# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

pingpong: pingpong.o
	${HIPCC} ${HIPFLAGS} pingpong.o -o pingpong

pingpong.o: pingpong.cpp
	${HIPCC} ${HIPFLAGS} -c pingpong.cpp

.PHONY: clean

clean:
	rm -f pingpong *.o
