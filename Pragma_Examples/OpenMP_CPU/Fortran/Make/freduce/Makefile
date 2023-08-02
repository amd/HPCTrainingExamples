FC               = gfortran
FFLAGS           = -O3 -g -fopenmp
LDFLAGS += -fopenmp
 
EXEC=freduce
 
all: $(EXEC)
 
%.o:%.F90
	$(FC) $(FFLAGS) -c $< -o $@
 
$(EXEC): freduce.o
	$(FC) -o $@ $^ $(LDFLAGS)
 
clean:
	rm -rf *.o $(EXEC)
