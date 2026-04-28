HIPCC = hipcc
CXXFLAGS = -O3 -Wall --offload-arch=gfx942
LIBS = -lrocsparse -lrocblas -lrocsolver

HEADERS = hip_utils.h matrix_utils.h preconditioner.h ic_preconditioner.h \
          jacobi_preconditioner.h asynch_jacobi_preconditioner.h pcg.h cg_cg.h \
          iterative_gauss_seidel.h

COMMON_OBJS = matrix_utils.o preconditioner.o ic_preconditioner.o \
              jacobi_preconditioner.o asynch_jacobi_preconditioner.o \
              iterative_gauss_seidel.o

PCG_OBJS = cg_driver.o $(COMMON_OBJS) pcg.o
CGCG_OBJS = cg_cg_driver.o $(COMMON_OBJS) cg_cg.o

all: run_cg run_cg_cg

%.o: %.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) -c -o $@ $<

run_cg: $(PCG_OBJS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LIBS)

run_cg_cg: $(CGCG_OBJS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -f *.o run_cg run_cg_cg
