# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

add_one: add_one.o
	${HIPCC} add_one.o -o add_one

add_one.o: add_one.cpp
	${HIPCC} -c add_one.cpp

.PHONY: clean

clean:
	rm -f add_one *.o
