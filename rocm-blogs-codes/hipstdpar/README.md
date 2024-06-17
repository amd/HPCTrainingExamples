# C++17 parallel algorithms and HIPSTDPAR #

The C++17 standard added the concept of [parallel algorithms][1] to the
pre-existing C++ Standard Library. The parallel version of algorithms like
[`std::transform`][2] maintain the same signature as the regular serial version,
except for the addition of an extra parameter specifying the
[`execution policy`][3] to use. This flexibility allows users that are already
using the [C++ Standard Library algorithms][4] to take advantage of multi-core
architectures by just introducing minimal changes to their code.

Starting with [ROCm][5] 6.1, the parallel algorithms seamlessly offload
to AMD accelerators via [HIPSTDPAR][6], as long as the user is willing to add an
extra compiler flag or two.

Whilst the functionality introduced by [HIPSTDPAR][6] is available for all AMD
GPUs (including consumer cards), this blog post focuses on the
[AMD CDNA2™ and CDNA3™ architectures][7] ([MI200][8] and [MI300][9] series
cards, respectively) using [ROCm][5] 6.1. As a code example, we focus on
the [Travelling Salesman Problem (TSP)][10] solver available [here][11].

## [The travelling salesman problem][10] ##

[The travelling salesman problem][10] tries to answer the following question:
"Given a list of cities and the distances between each pair of cities, what is
the shortest possible route that visits each city exactly once and returns to
the origin city?". This problem is particularly hard to solve (NP-hard) due to
exponential complexity; adding an extra city to the list causes an exponential
growth in the number of combinations to check. Solving this problem by just
enumerating all possible combinations and checking each one of them is
computationally prohibitive for problems with more than 17 or 18 cities. For real
world applications, advanced methods are used (cutting planes and branch and
bound techniques) but for the purposes of this blog we focus on a
embarrassingly parallel implementation of the brute-force approach.

The [TSP][10] solver we look at relies on the following function to check
the various permutations of cities and pick the one with the lowest
cost/distance. Here is a detailed implementation that does not make use of any
parallelism:

```cpp
template<int N>
route_cost find_best_route(int const* distances)
{
  return std::transform_reduce(
    counting_iterator(0),
    counting_iterator(factorial(N)),
    route_cost(),
    [](route_cost x, route_cost y) { return x.cost < y.cost ? x : y; },
    [=](int64_t i) {
    int cost = 0;

    route_iterator<N> it(i);

    // first city visited
    int from = it.first();

    // visited all other cities in the chosen route
    // and compute cost
    while (!it.done())
    {
      int to = it.next();
      cost += distances[to + N*from];
      from = to;
    }

    // update best_route -> reduction
    return route_cost(i, cost);
  });
}
```

The [`std::transform_reduce`][12] algorithm performs two operations:

1. a transformation (equivalent to a map operation) implemented by the lambda
   function passed as final argument;
2. a reduction operation, expressed as lambda function as fourth argument.

The function above runs through all elements from `0` to `N!`, each of which
expresses a particular permutation of all cities, computes the cost of the
particular path, and returns an instance of `route_cost` object that includes
the id of the particular path and the cost associated with it. At the end, a
reduction is performed by comparing the cost of the various paths and selecting
the one with lowest cost.

On an [AMD Zen4 processor][13], this serial code takes about 11.52 seconds to
compute the best path for a [TSP][10] instance involving twelve cities. The
same code takes about 156 seconds for a [TSP][10] instance involving thirteen
cities. This is a normal consequence of the exponential growth of the search
space imposed by the [TSP][10].

## Execution policies and [HIPSTDPAR][6] ##

Since each of the `N!` paths are independent, computing their individual cost is
an embarrassingly parallel operation. C++17 allows developers to easily
parallelize the previous code by just passing an [`execution policy`][3] as the
first argument of the algorithm invocation. The C++17 standard defines three
possible execution policies:

- `std::execution::sequenced_policy` and the corresponding policy object to pass
  as argument `std::execution::seq`
- `std::execution::parallel_policy` and the corresponding policy object to pass
  as argument `std::execution::par`
- `std::execution::parallel_unsequenced_policy` and the corresponding policy
  object to pass as argument `std::execution::par_unseq`

Execution policies allow the user to convey information to the implementation
about the invariants that user code shall enforce / maintain, thus allowing the
latter to possibly adopt more favourable / performant execution.

### `std::execution::sequenced_policy` ###

The sequenced policy constrains the implementation to perform all operations on
the thread that invoked the algorithm, inhibiting possible parallel execution.
All operations are indeterminately sequenced within the caller thread, which
implies subsequent invocations of the same algorithm, within the same thread,
can have their operations sequenced differently.

### `std::execution::parallel_policy` ###

The parallel policy allows the implementation to employ parallel execution.
Operations may be performed on the thread that invoked the algorithm or on
threads created by the standard library implementation. All operations are
indeterminately sequenced within a thread, for all threads used to perform the
computation described by the algorithm invocation. Furthermore, there are no
ordering guarantees provided for the element access function invocations
themselves. Compared to the sequenced policy, additional constraints are imposed
on the various components used by the algorithm. In particular, operations on
iterators, values, and callable objects, as well as their transitive closure,
must be data race free.

In the previous example, it is possible to parallelize the `find_best_route`
function by passing as first extra argument the `std::execution:par` policy as
follows:

```cpp
return std::transform_reduce(
  std::execution::par, // THE SIMPLE CHANGE
  counting_iterator(0),
  counting_iterator(factorial(N)),
  route_cost(),
  [](route_cost x, route_cost y) { return x.cost < y.cost ? x : y; },
  [=](int64_t i)
```

By making this simple change, the code will now run on all CPU cores available.
On the CPU portion of a [MI300A][14], equipped with 48 [Zen4][13] logical cores,
solving an instance of [TSP][10] with 12 cities takes about 0.34 seconds.
This parallel run is almost 34x faster compared to the 11.52 seconds needed by the serial version!
For an instance of [TSP][10] with thirteen cities the parallel version takes
about 5 seconds.
Finally, for a bigger problem involving fourteen cites, the 48 [Zen4][13]
logical cores take about 77 seconds.

### `std::execution::parallel_unsequenced_policy` ###

This policy guarantees the most restrictive requirements are met by user
provided code. An algorithm invoked with this policy may perform the steps in
unordered and unsequenced ways with respect to one another. This means that the
various operations can be interleaved with each other on the same thread. Also
any given operation may start on a thread and end on a different thread. When
specifying the parallel unsequenced policy, the user guarantees that no
operations that entail calling a function that `synchronizes-with` another
function are employed. In practice, this means that user code does not do any
memory allocation / deallocation, only relies on lock-free specializations of
[`std::atomic`][15], and does not rely on synchronization primitives such as
[`std::mutex`][16].

**_This policy is currently the only one that can be chosen to offload_**
**_parallelism to AMD accelerators_**. To trigger the GPU offload of **_all_**
parallel algorithms invoked with the parallel unsequenced policy, the
[`--hipstdpar`][19] flag must be passed at compile time. Furthermore, for GPU
targets other than the current default ([`gfx906`][40]), the user must also pass
[`--offload-arch=`][41] specifying which GPU is being used.

On [MI300A][14], by simply switching policy and recompiling with the
aforementioned flags, the execution time for an instance of [TSP][10] with
thirteen cities goes down to 0.5 seconds. When using fourteen cities, the use of
the GPU portion of [MI300A][14] brings down the execution time to 4.8 seconds
from the 77 seconds needed by the parallel version running on 48 [Zen4][13]
logical cores. And because everybody loves a good table, let us conclude this
section by summarising the progression from sequenced execution on the CPU
to parallel unsequenced execution offloaded to the accelerator:

| 14-city TSP        | Timing (s) |
|:------------------:|:----------:|
| `seq`              | 2337       |
| `par`              | 77         |
| `par_unseq` on CPU | 75         |
| `par_unseq` on GPU | 4.8        |

## [TeaLeaf][43] ##

A more complex example showing the use and performance of [HIPSTDPAR][6] is
[TeaLeaf][43]. The code is a C++ implementation of the
[TeaLeaf heat conduction mini-app][44] from the University of Bristol, UK.
Multiple implementations illustrate various parallel programming paradigms,
including [HIP][30] and parallelised standard algorithms. This allows us to make
a fair performance comparison between an optimized, [HIP][30]-based
implementation and a [HIPSTDPAR][6] one. For the purpose of this test, we
selected the `tea_bm_5.in` benchmark, comprising of a 2D grid of 4000x4000 cells
and 10 time steps.

For the [HIPSTDPAR][6] version, on a [MI300A][14] card, the following output
is obtained:

```bash
Timestep 10
CG:                    3679 iterations
Wallclock:             40.884s
Avg. time per cell:    2.555271e-06
Error:                 9.805532e-31

Checking results...
Expected 9.546235158221428e+01
Actual   9.546235158231138e+01
This run PASSED (Difference is within 0.00000000%)
```

As for the [HIP][30] version, it performs as follows:

```bash
Timestep 10
CG:                    3679 iterations
Wallclock:             34.286s
Avg. time per cell:    2.142853e-06
Error:                 9.962546e-31

Checking results...
Expected 9.546235158221428e+01
Actual   9.546235158231144e+01
This run PASSED (Difference is within 0.00000000%)
```

The performance difference in-between the two versions stems from the overhead
associated with handling the initial page-in of non-resident memory. To "even
things out", the HIP version can be adjusted to use [`hipMallocManaged()`][27]
as well, instead of [`hipMalloc()`][45]. This particular configuration is
already available in the [HIP][30] version of [TeaLeaf][43] and it can be
enabled by passing a simple flag at compile time. The following is the output
for the [HIP][30] version of [TeaLeaf][43] when using
[`hipMallocManaged()`][27] and [XNACK][21] for all GPU allocations.

```bash
Timestep 10
 CG:                    3679 iterations
 Wallclock:             39.573s
 Avg. time per cell:    2.473331e-06
 Error:                 9.962546e-31

 Checking results...
 Expected 9.546235158221428e+01
 Actual   9.546235158231144e+01
 This run PASSED (Difference is within 0.00000000%)
```

As expected, the performance of the [HIP][30] version when introducing
[`hipMallocManaged()`][27] is comparable with the one observed for the
[HIPSTDPAR][6] version. In closing, we will note that ongoing work is expected
to reduce the overhead, thus bringing the offloaded version closer to the
[HIP][30] one.

## Nuts and bolts of [HIPSTDPAR][6] ##

The ability to offload C++ Standard Parallel algorithm execution to the GPU
relies on the interaction between the [LLVM compiler][17], [HIPSTDPAR][6], and
[`rocThrust`][18]. Starting from [ROCm][5] 6.1, the LLVM compiler used to
compile regular HIP codes will be able to forward invocations of standard
algorithms which take the `parallel_unsequenced_policy` execution policy to the
[HIPSTDPAR][6] header-only library when the [`--hipstdpar`][19] flag is passed.
The header-only library is in charge of mapping the parallel algorithms used by
C++ Standard Library into the equivalent [`rocThrust`][18] algorithm invocation.
This very simple design allows for a low overhead implementation of the
offloading for parallel standard algorithms. A natural question to ask at this
point is: "computation is nice but what about the memory it operates on?". By
default, [HIPSTDPAR][6] assumes that the underlying system is
[HMM (Heterogeneous Memory Management)][20]-enabled, and that page migration is
possible via the handling of retry-able page-faults implemented atop
[XNACK][21] (e.g., `export HSA_XNACK=1`). This particular mode is referred to as
**[HMM][20] Mode**.

When these two requirements are satisfied, code offloaded to the GPU
(implemented via [`rocThrust`][18]) triggers the page migration mechanism
and data will automatically migrate from host to device. On [MI300A][14],
although physical migration is neither needed nor useful, handling page faults
via [XNACK][21] is still necessary. For more details about page migration please
refer to the following [blog post][22].

On systems without [HMM][20] / [XNACK][21] we can still use [HIPSTDPAR][6] by
passing an extra compilation flag: [`--hipstdpar-interpose-alloc`][23]. This
flag will instruct the compiler to replace **_all_** dynamic memory allocations
with compatible [`hipManagedMemory`][24] allocations implemented in the
[HIPSTDPAR][6] header-only library. For example, if the application being
compiled, or one of its transitive inclusions, allocates free store memory via
[`operator new`][25], that call will be replaced with a call to
`__hipstdpar_operator_new`. By looking at the implementation of that function in
the [HIPSTDPAR library][26] we see that the actual allocation is performed via
the [`hipMallocManaged()`][27] function. By doing so on a non [HMM][20]-enabled
system, host memory is pinned and directly accessible by the GPU without
requiring any page-fault driven migration to the GPU memory. This particular
mode is referred to as **Interposition Mode**.

### Restrictions ###

For both [HMM][20] and Interposition modes, the following restrictions apply:

1. Pointers to function, and all associated features, such as e.g. dynamic
   polymorphism, cannot be used (directly or transitively) by the user provided
   callable passed to an algorithm invocation;
2. Global / namespace scope / `static` / `thread` storage duration variables
   cannot be used (directly or transitively) in name by the user provided
   callable;

   - When executing in **[HMM][20] Mode** they can be used in address e.g.:

     ```cpp
     namespace { int foo = 42; }

     bool never(const vector<int>& v) {
       return any_of(execution::par_unseq, cbegin(v), cend(v), [](auto&& x) {
         return x == foo;
       });
     }

     bool only_in_hmm_mode(const vector<int>& v) {
       return any_of(execution::par_unseq, cbegin(v), cend(v),
                     [p = &foo](auto&& x) { return x == *p; });
     }
     ```

3. Only algorithms that are invoked with the `parallel_unsequenced_policy` are
   candidates for offload;
4. Only algorithms that are invoked with iterator arguments that model
   [`random_access_iterator`][28] are candidates for offload;
5. [Exceptions][29] cannot be used by the user provided callable;
6. Dynamic memory allocation (e.g. [`operator new`][25]) cannot be used by the
   user provided callable;
7. Selective offload is not possible i.e. it is not possible to indicate that
   only some algorithms invoked with the `parallel_unsequenced_policy` are to
   be executed on the accelerator.

In addition to the above, using **Interposition Mode** imposes the following
additional restrictions:

1. All code that is expected to interoperate has to be recompiled with the
   [`--hipstdpar-interpose-alloc`][23] flag i.e. it is not safe to compose
   libraries that have been independently compiled;
2. automatic storage duration (i.e. stack allocated) variables cannot be used
   (directly or transitively) by the user provided callable e.g.

   ```cpp
   bool never(const vector<int>& v, int n) {
     return any_of(execution::par_unseq, cbegin(v), cend(v),
                   [p = &n](auto&& x) { return x == *p; });
   }
   ```

## But why? ##

After what has been something of a whirlwind tour, it is not unreasonable to
ask "but how does this benefit me, the C++ developer?".
The goal of [HIPSTDPAR][6] is to allow any C++ developer that is employing
standard algorithms to leverage GPU acceleration with no cognitive overload. The
application developer can remain firmly planted in the Standard C++ world, without having to step into
the brave new world of GPU specific languages such as e.g. [HIP][30] or
[SYCL][31]. Fortunately for us, our particular example allows for some limited, quantitative
insight into just how close we got to this goal.  The Tealeaf author has implemented
the solver via multiple programming interfaces which means that we can use the
[`cloc`][32] tool to count the lines of code needed by the `tsp.cpp`
implementation:

| Programming Interface                      | LoC |
|:------------------------------------------:|:---:|
| [Kokkos][33]                               | 145 |
| [OpenACC][34]                              | 142 |
| [OpenMP][35]                               | 116 |
| Standard C++ Serial                        | 112 |
| Standard C++ Parallel Algorithms           | 107 |
| [SYCL][31]                                 | 169 |

It is apparent that using compiler flag driven offload, as enabled by
[HIPSTDPAR][6], saves on a considerable amount or typing - up to 57% versus
[SYCL][31], for example. This enables a more natural journey towards GPU
accelerated execution. As a result, the programmer can focus on the algorithm /
problem solving, at least initially, and discover generic algorithmic
optimisations that are profitable for the GPU, without having to dive head-first
into GPU "arcana".

## TL;DR, just tell me how to go fast ##

Initially, [HIPSTDPAR][6] is officially supported on Linux, with Windows support
forthcoming at a future date. Starting from
[an environment that has been set up for ROCm][36], using the package manager to
install the `hipstdpar` package will, in general, bring in all the required
goodness. Additionally, at the time of writing, a dependency on [TBB][37]
exists, as a consequence of standard library implementation details (see e.g.
[Note 3][38]). Therefore, it is necessary to install the system's [TBB][37]
package (e.g. [`libtbb-dev`][39] on Ubuntu). Armed thusly, and assuming that we
have a `main.cpp` file which uses some standard algorithms to solve a given
problem, the compiler driver invocation:

```bash
clang++ --hipstdpar main.cpp -o main
```

transparently offloads all algorithm invocations that use the
`std::execution::parallel_unsequenced_policy` execution policy, if we are
targeting a GPU compatible with the [`gfx906`][40] ISA (i.e. [Vega20][42]).
Otherwise, we also have to specify the target for offload:

```bash
clang++ --hipstdpar --offload-arch=gfx90a main.cpp -o main
```

## Conclusion ##

In this post, we provided a high level overview of the [ROCm][5] support for
offloading [C++ Standard Parallel Algorithms][4], aiming to show how existing
C++ developers can leverage GPU acceleration without having to adopt any new,
GPU specific, language (e.g., [HIP][30]) or directives (e.g., [OpenMP][35]).

We believe that this standard, extremely accessible, way of exploiting hardware
parallelism will be particularly beneficial for applications targeting
[MI300A][14] accelerators, where the CPU and the GPU share the same pool of HBM.
Although not demonstrated today, the combination of the APU architecture and
[HIPSTDPAR][6] can enable fine-grained cooperation between CPU and GPU, which
become true peers, accessible via a uniform programming interface.

For an in-depth look at the compiler side of [HIPSTDPAR][6] support, the
interested reader should peruse the associated [AMD-LLVM][46] documentation.

If you have any questions please reach out to us on GitHub
[Discussions](https://github.com/amd/amd-lab-notes/discussions).

[1]: https://en.cppreference.com/w/cpp/experimental/parallelism
[2]: https://en.cppreference.com/w/cpp/algorithm/transform
[3]: https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t
[4]: https://en.cppreference.com/w/cpp/algorithm
[5]: https://rocm.docs.amd.com/en/latest/
[6]: https://github.com/ROCm/roc-stdpar
[7]: https://www.amd.com/en/technologies/cdna.html
[8]: https://www.amd.com/en/products/accelerators/instinct/mi200.html
[9]: https://www.amd.com/en/products/accelerators/instinct/mi300.html
[10]: https://en.wikipedia.org/wiki/Travelling_salesman_problem
[11]: https://github.com/pkestene/tsp
[12]: https://en.cppreference.com/w/cpp/algorithm/transform_reduce
[13]: https://en.wikipedia.org/wiki/Zen_4
[14]: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html
[15]: https://en.cppreference.com/w/cpp/atomic/atomic
[16]: https://en.cppreference.com/w/cpp/thread/mutex
[17]: https://www.llvm.org/
[18]: https://github.com/ROCm/rocthrust
[19]: https://releases.llvm.org/18.1.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-hipstdpar
[20]: https://www.kernel.org/doc/html/latest/mm/hmm.html
[21]: https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html#xnack
[22]: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-mi200-memory-space-overview/#enabling-page-migration
[23]: https://releases.llvm.org/18.1.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-hipstdpar-interpose-alloc
[24]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory_m.html
[25]: https://en.cppreference.com/w/cpp/memory/new/operator_new
[26]: https://github.com/ROCm/roc-stdpar/blob/5c7f44d8418273671671378871afc00fddc4aa14/include/hipstdpar_lib.hpp#L82
[27]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory_m.html#gaadf4780d920bb6f5cc755880740ef7dc
[28]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
[29]: https://en.cppreference.com/w/cpp/language/exceptions
[30]: https://rocm.docs.amd.com/projects/HIP/en/latest/index.html
[31]: https://www.khronos.org/sycl/
[32]: https://github.com/AlDanial/cloc
[33]: https://github.com/kokkos/kokkos
[34]: https://www.openacc.org/
[35]: https://www.openmp.org/
[36]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick
[37]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html
[38]: https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html#status.iso.2017
[39]: https://packages.ubuntu.com/focal/libtbb-dev
[40]: https://llvm.org/docs/AMDGPUUsage.html#id14
[41]: https://releases.llvm.org/18.1.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-offload-arch
[42]: https://www.techpowerup.com/gpu-specs/amd-vega-20.g848
[43]: https://github.com/UoB-HPC/TeaLeaf
[44]: https://github.com/UK-MAC/TeaLeaf
[45]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html#ga4c6fcfe80010069d2792780d00dcead2
[46]: https://github.com/ROCm/llvm-project/blob/amd-mainline-open/clang/docs/HIPSupport.rst#c-standard-parallelism-offload-support-compiler-and-runtime
