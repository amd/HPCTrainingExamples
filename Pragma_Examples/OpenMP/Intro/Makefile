# targetdata.cpp gives wrong answer
# saxpy6 crashes during execution
all: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy2f saxpy8 targetdata

# From slide 3 -- CPU version
saxpy_cpu: saxpy_cpu.cpp
	amdclang++ -g -O3 -fopenmp saxpy_cpu.cpp -o saxpy_cpu

# From slide 8 but without map clause
saxpy1: saxpy1.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy1.cpp -o saxpy1

# From slide 8 with map clause
saxpy2: saxpy2.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy2.cpp -o saxpy2

# From slide 9, Fortran example
# .f90 or .F90 extension sets code to free form
# some additional code has been added to time the loop and a test
# is added to avoid compilers from optimizing out the operation
# Capitalization of file extension, .F90, turns on preprocessing
saxpy1f: saxpy1f.F90
	amdflang -g -O3 -Mfreeform -Mpreprocess -fopenmp --offload-arch=native saxpy1f.F90 -o saxpy1f

# From slide 10 with optimized map clause
saxpy3: saxpy3.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy3.cpp -o saxpy3

# From slide 11 using subroutine call with pointers
saxpy4: saxpy4.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy4.cpp -o saxpy4

# From slide 14 adding parallel for simd
# simd is not recognized by AMD and Nvidia, but others such as Intel still use it
# parallel construct is a triplet of "parallel for simd"
saxpy5: saxpy5.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy5.cpp -o saxpy5

# From slide 16 adding parallel for simd
saxpy6: saxpy6.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy6.cpp -lstdc++ -o saxpy6

# From slide 17
# adding teams distribute to hardware levels
# hardware levels is a triplet of "target teams distribute"
saxpy7: saxpy7.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy7.cpp -o saxpy7

# From slide 17 -- Fortran example
saxpy2f: saxpy2f.F90
	amdflang -g -O3 -Mfreeform -Mpreprocess -fopenmp --offload-arch=gfx90a saxpy2f.F90 -o saxpy2f

# From slide 19
saxpy8: saxpy8.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native saxpy8.cpp -o saxpy8

# From slide 22
targetdata: targetdata.cpp
	amdclang++ -g -O3 -fopenmp --offload-arch=native targetdata.cpp -o targetdata

run: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy8 targetdata
	./saxpy_cpu
	./saxpy1
	./saxpy2
	./saxpy1f
	./saxpy3
	./saxpy4
	./saxpy5
	#./saxpy6
	./saxpy7
	./saxpy8
	./targetdata

clean:
	rm -f saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy5a saxpy6 saxpy7 saxpy2f saxpy8 targetdata *.o
