# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

stencil: stencil.o
	${HIPCC} ${HIPFLAGS} stencil.o -o stencil

stencil.o: stencil.cpp
	${HIPCC} ${HIPFLAGS} -c stencil.cpp

.PHONY: clean

clean:
	rm -f stencil *.o
