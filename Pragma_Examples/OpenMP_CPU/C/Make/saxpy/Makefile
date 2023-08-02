CC               = gcc
CFLAGS           = -O3 -g -std=c99 -fopenmp
LDFLAGS += -fopenmp

EXEC=saxpy 

all: $(EXEC)

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

codelet.o:codelet.c
	$(CC) $(CFLAGS) -c $< -o $@

$(EXEC): codelet.o main.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf *.o $(EXEC) *.bin
