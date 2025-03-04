# Solve a Linear System of ODEs with ROCm Libraries

This simple example will show how to use some of the available ROCm libraries to solve a system of linear ordinary differential equations (ODEs).
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

In this directory, there is code that will use the `rocblas` library to compute the above approximation of $\exp^{At}$. The code can be easily modified to be extended for any matrix, but since we want to validate our results, we will be considering a case for which an analytic solution is available, so we can measure the error of the approximate solution, which will be given by:

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

