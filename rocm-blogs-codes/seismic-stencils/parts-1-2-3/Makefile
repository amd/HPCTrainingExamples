HIPC=hipcc
CFLAGS=-O3 -g
ARCH=gfx90a
radius=4
vec=2
HIPFLAGS=-O3 -g --offload-arch=$(ARCH) -std=c++17 --save-temps -DRADIUS=$(radius) -DVEC_EXP=$(vec)

all: stencils

sliding_window : sliding_window.cpp
	$(HIPC) $(CFLAGS) $< -o $@.x

stencils : stencils.cpp
	$(HIPC) $(HIPFLAGS) $< -o $@_R_$(radius)_vec_$(vec).x

clean: clean-temps clean-profiling
	rm -rf *.o *.x

clean-temps:
	rm -f *-hip-* *-host-*

clean-profiling:
	rm -f results* *.csv
