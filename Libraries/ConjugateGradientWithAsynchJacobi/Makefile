HIPCC = hipcc
CXXFLAGS = -O3 -Wall
LIBS = -lrocsparse -lrocblas -lrocsolver

HEADERS = hip_utils.h matrix_utils.h preconditioner.h ic_preconditioner.h \
          jacobi_preconditioner.h asynch_jacobi_preconditioner.h pcg.h

OBJS = cg_driver.o matrix_utils.o preconditioner.o ic_preconditioner.o \
       jacobi_preconditioner.o asynch_jacobi_preconditioner.o pcg.o

TARGET = run_cg

all: $(TARGET)

%.o: %.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -f *.o $(TARGET)
