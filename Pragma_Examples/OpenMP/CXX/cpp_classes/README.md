# OpenMP Offloading for C++ Codes that use Classes

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/cpp_classes` from the Training Examples repository

These examples show how to use OpenMP for GPU offloading in the context of a C++ code that makes uses of classes, and a programming paradigm where the most relevant members of the class are private, with their associated values accessed and modified by appropriate `get` and `set` functions.

In the present directory, you will find two subdirectories, one called `usm` and one called `explicit`. 

## The `usm` Sub-directory

In this context, `usm` stands for unified shared memory, which is what we are requiring for the code samples in this directory. 
To compile the code in the `usm` directory, do:

```
module load rocm
module load amdclang
export HSA_XNACK=1
make
```

If the `amdclang` module is not available on your system, make sure to do:
```
export CXX=$ROCM_PATH/llvm/bin/amdclang++
```
before running the `make` command.

Note that if one was to not set `HSA_XNACK=1` the code would not compile, because we are requiring unified shared memory with the following pragma line in `main.cpp`:

```
#pragma omp requires unified_shared_memory
```

You may have noticed many compiler warnings such as this one:

```
main.cpp:22:20: warning: Type 'daxpy' is not trivially copyable and not guaranteed to be mapped correctly [-Wopenmp-mapping]
   22 |       double val = data.getConst() * data.getX(i) + data.getY(i);
```

From the warning, you can already see what potential issues can arise in a C++ programming paradigm like the one we decided to set ourselves in.
Luckily, we can get around those warnings thanks to the use of unified shared memory.

In the `usm` directory, there are two subdirectories, `daxpy` and `operations`.

### The `daxpy` Sub-directory

Here we are defining a class object to perform a daxpy operation. Notice that the daxpy operation is actually performed within the `main.cpp` and not through a member function like it will be for the `explicit` case. Moreover, we are using the `get` and `set` member functions of the daxpy class from within the target region without incurring in any memory access issue, thanks to the unified shared memory framework.

### The `operations` Sub-Directory

The code in the `operations` directory adds one layer of complexity and performs a daxpy from the `main.cpp` file but using a class called `operations` that has two more classes as members: the `daxpy` class already mentioned before, and the `norm` class, which will compute a user-defined norm of the output of the daxpy operation. Note that everything works seamlessly even when calling member functions from the `ops` object: these member functions are wrappers to the member functions of the `daxpy` and `norm` class members.

## The `explicit` Sub-directory

This sub-directory contains example code that is meant to work even without enabling unified shared memory, meaning that it will compile and run regardless of whether `HSA_XNACK=1`. To compile:

```
module load rocm
module load amdclang
make
```

Again, make sure that the CXX environment variable is set as below, before running the `make` command:
```
export CXX=$ROCM_PATH/llvm/bin/amdclang++
```

The directory is named `explicit` because we are explicitly taking care of all the data movement between host and device, helping the compiler with figuring out how to perform the offload to GPU. The only sub-directory here is `daxpy`.

### The `daxpy` Sub-directory

The explicit memory movement scenario gets tricky really quickly, as you have seen with the numerous warning messages produced by the compiler when building the `usm` examples. Things get particularly complicated when using anything that is not just a pointer for our data members, such as for instance standard vectors. It is non-trivial to handle calls of members functions within a target region, and in fact in this case you can see that the daxpy operation is actually performed by a member function called `apply` that is called from `main.cpp`. In the `daxpy.hpp` file where the `daxpy` class is declared, we have now included in the constructor the following pragma:

```bash
#pragma omp target enter data map(to: x_[0:N_],y_[0:N_], N_, a_)
```

The above pragma creates a data environment for an unstructured data region and maps x_, y_, N_ and a_ to the device. Note that we also had to explicitly map the scalars to make sure that they are available on the device when we call the `apply` function, which is defined in `daxpy.cpp`. The following pragma is included in the destructor for the class:

```bash
#pragma omp target exit data map(delete: x_[0:N_],y_[0:N_], N_, a_)
```

