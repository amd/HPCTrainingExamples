## C++ member function

The first example is where there is a compute method in the Science class that is called from a parallel target region.

```
cd 1_member_function
```

The original code is shown in `0_member_function_portyourself`

```
cd 0_member_function_portyourself
```

Try adding the `#pragma omp target teams loop` directive to the loop in the `bigscience.cc` routine to port it to run on the device.

To see the solution to the porting, see the code in `1_member_function` directory.

Looking at the loop in `bigscience.cc`:

```
#pragma omp target teams loop
   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }
```

To try out the code, compile it and run it.

```
make
./bigscience
```

Note that nothing needs to be done to the class in `Science.hh`. Why is this? Basically, the defined method function in `Science.hh`
is in-lined into the `bigscience.cc` file. So it is handled by the directive added around the loop in `bigscience.cc`.

## C++ member function external

So what happens when the compute method is defined in a different file? For this case, let's take a look at the next example
in `2_method_function_external`.

```
cd ../2_method_function_external
```

Try porting the code in `0_member_function_external_portyourself`. Note that in this example, the compute member function is defined
in `Science_member_functions.cc`

For the solution, go to the `1_member_function_external` directory

```
cd ../1_member_function_external
```

Note that now we have to add `#pragma omp declare target` around the compute method definition. We also need a 
`#pragma omp end declare target` directive to close out the declare target region.

Let's try compiling and running the example

```
make
./bigscience
```

The next example, `2_member_function_external_data` uses a data value `init_value` from the Science class. The thing
to note is that we do not need to add a `#pragma omp declare target` around the declaration in the class.

Check that this runs fine with your compiler

```
cd ../2_member_function_external_data
make
./bigscience
```

## C++ virtual methods

Additional complexity in C++ classes can cause difficulties with porting to GPUs. Fundamentally, the GPU language is C with only
a little support for C++. So let's take a look at a simple virtual method where class inheritance is used. 

```
cd ../../3_virtual_methods
```

The original CPU C++ code is given in `0_virtual_methods_portyourself`. We create a new HotScience class that is based on the Science class. 
The new class is defined in `HotScience.hh`. It overrides the compute method. The method definition for the new compute function is 
in `HotScience_member_functions.cc`.

First, let's verify that the original code works.

```
cd 0_virtual_methods_portyourself
make
./bigscience
```

Try porting this version and see what might be required. 

The solution is given in `1_virtual_methods` directory.

```
cd ../1_virtual_methods
```

Examine the source code files to see what is needed. Note that now the `#pragma omp declare target` block is needed around the 
method definition in `HotScience_member_functions.cc`. Let's verify that this works with your current compiler.

```
make
./bigscience
```

A special note here for the current amdclang++ compiler. With the changes to the source code, the compiler issues a warning
about maybe not being mapped correctly

```
warning: type 'HotScience' is not trivially copyable and not guaranteed to be mapped correctly
```

The code still compiles and runs properly. To suppress the warning, `-Wno-openmp-mapping` has been added to CXXFLAGS in the
Makefile.
