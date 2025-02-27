# Solve a Linear System of ODEs with ROCm Libraries

This simple example will show how to use some of the available ROCm libraries to solve a simple system of linear ordinary differential equations (ODEs).
These systems are solved to describe, for instance, the evolution of dynamical systems, wherthe velocity of the system at a given time depends linearly on the position. The term linear here is intended in the mathematical sense.

## Problem Definition

Consider the initial value problem:

$$
\dot{\mathbf{x}} = A \mathbf{x} \qquad 
\mathbf{x}(0) = 
\begin{pmatrix}
1 \\
0
\end{pmatrix},
$$
 
where

$$
A=
\begin{pmatrix}
-2 & -1 \\
1 & -2
\end{pmatrix}, \qquad
\mathbf{x}_0=
\begin{pmatrix}
1 \\
0
\end{pmatrix}.
$$

By the fundamental theorem of linear systems, the solution $\mathbf{x}(t)$ at any time $t$ is can be found analytically, and is given by:

$$
\mathbf{x}(t)=\exp^{At}\mathbf{x}=\exp^{-2t}
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

The matrix exponential $\exp^{At}$, which in this case is known analytically and is equal to:

$$
\exp^{At}=\exp^{-2t}
\begin{pmatrix}
\cos(t) & - \sin(t) \\
\sin(t) & \cos(t)
\end{pmatrix}
$$

can actually be approximated using a truncated series, which is an infinite summation truncated to a finite number of terms, namely:

$$
\exp^{At} \approx \sum_{k=0}^N\dfrac{A^k t^k}{k!}.
$$

Note that as $N$ goes to infinity, the above becomes an actual equality. In this example, we will approximate numerically the matrix exponential using the above, and verify that the error becomes smaller as more terms are included. The error is defined as the norm of the exact solution minus the approximate solution obtained with the approximation of the matrix exponential, namely:

$$
\Delta_N=\|\mathbf{x}(t)-\mathbf{y}_N(t)\|,
$$

where

$$
\mathbf{y}_N(t)=(\sum_{k=0}^N\dfrac{A^k t^k}{k!})\mathbf{x}_0.
$$

Remember, what is inside the parenthesis above is a matrix.

## Implementation

The implementation of the above mathematical problem will be carried out with the help of several ROCm libraries. For instance, the matrix products involved in the truncated series will be carried out with `rocBLAS`.
