# Solve a Linear System of ODEs with ROCm Libraries

This is HPCTrainingExamples/Libraries/matrix_exponential/README.md in the training examples repository.

This example shows how to use the `rocBLAS` library to solve a system of linear ordinary differential equations (ODEs).
These systems are used to describe, for instance, the evolution of dynamical systems, where the velocity of the system at a given time depends linearly on the position. The term linear here is intended in the mathematical sense.

## Problem Definition

Consider the initial value problem:

$$
\dot{x}(t) = A x(t) \qquad
\mathbf{x}(0) = x_0,
$$

where $A$ is and $n$ by $n$ matrix and $x_0$ is a vector in $\mathbb{R}^n$, the initial value of the system.
By the fundamental theorem of linear systems, the above system has a unique solution given by:

$$
x(t)=\exp^{At}x_0.
$$

The matrix $\exp{At}$ is called matrix exponential. The following identity holds:

$$
\exp^{At} = \sum_{k=0}^{\infty}\dfrac{A^k t^k}{k!}.
$$

The above infinite series is absolutely convergent and can be used to obtain an approximation of the matrix exponential, by a truncation to a finite number of terms $N$:

$$
\exp^{At} \approx \sum_{k=0}^N\dfrac{A^k t^k}{k!}.
$$

In this directory, there is code that will use the `rocBLAS` library to compute the above approximation of $\exp^{At}$. The code can be easily modified to be extended for matrices of arbitrary size, but since we want to validate our results, we will be considering a 2x2 case for which an analytic solution is available, so we can measure the error of the approximate solution, which will be given by:

$$
\Delta_N(t)=\|x(t)-y_N(t)\|_2,
$$

where

$$
y_N(t)=\sum_{k=0}^N\dfrac{A^k t^k}{k!}x_0
$$

is the approximate solution obtained with a truncation of the infinite series to $N$ terms. Remember, what multiplies $x_0$ above is a matrix.

Hence, the case that we consider will have $A$ and $x_0$ define by:

$$
A=
\begin{pmatrix}
-2 & -1 \\
1 & -2
\end{pmatrix}, \qquad
x(0)=
\begin{pmatrix}
1 \\
0
\end{pmatrix}.
$$

By the fundamental theorem of linear systems, the analytical solution $x(t)$ at any time $t$ for the above choice of $A$ and $x_0$ is given by:

$$
x(t)=\exp^{At}x=\exp^{-2t}
\begin{pmatrix}
\cos(t) & - \sin(t) \\
\sin(t) & \cos(t)
\end{pmatrix}
\begin{pmatrix}
1 \\
0
\end{pmatrix}
=\exp^{-2t}
\begin{pmatrix}
\cos(t) \\
\sin(t)
\end{pmatrix}
$$

## Compile and Run

There are currently two directories: `device_sync` and `streams_sync`: the main difference between the code in these two directories is that in `device_sync` we are calling `hipDeviceSynchronize()` after each `rocblas_dgemm` call whereas in `streams_sync` we are creating streams for each OpenMP CPU thread and then synchronizing with `hipStreamSynchronize(stream)`. In `stream_sync` there are two sub-directories, one called `hip` and one called `interop`. In the one called `hip` we are manually setting the number of OpenMP CPU threads to 4, and then creating 4 streams for each OpenMP CPU thread. This is done before the `#pragma omp parallel for ...` that creates the threads. In the `interop` sub-directory, we are using the OpenMP `interop` directive to create a foreign (to OpenMP) synchronization object (a HIP stream) from within the parallel OpenMP loop and then set the execution of rocblas on this specific stream. Note that in this way the OpenMP threads are using more than one stream since a new stream is created during each iteration of the parallel for loop. 

To compile and run do (this applies to both directories):

```
module load rocm
module load amdclang
export HSA_XNACK=1
make
./mat_exp
```

If the `amdclang` module is not available in your system, after loading the ROCm module, do `export CXX=$(which amdclang++)`.
Setting the environment variable `HSA_XNACK=1` enables managed memory and is necessary to not incur in memory access faults, as we are passing host pointers to the `rocblas_dgemm` call.

## Comment on Performance

Note that the code as presented in the `device_sync` and `streams_sync` directories is intended to show a simple application of rocBLAS and its performance is not fully optimized. A size of 2x2 for the matrices is normally too small to justify offloading to the GPU but, as clarified above, the choice of a 2x2 size is to allow validation of the code. A generalization to arbitrary size and tuned performance is outside the scope of these examples and is left to the interested users.
