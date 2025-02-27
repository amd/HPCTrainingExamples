# Solve a Simple ODE with ROCm Libraries

This simple example will show how to use some of the available ROCm libraries to solve a simple system of linear ordinary differential equations (ODEs).
These systems are solved to describe, for instance, the evolution of dynamical systems, wherthe velocity of the system at a given time depends linearly on the position. The term linear here is intended in the mathematical sense.

## Problem Definition

Consider the initial value problem:

$$
\dot{\bm{x}} = A \bm{x}
\bm{x}(0) = 
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
\end{pmatrix}.
$$

By the fundamental theorem of linear systems, the solution $\bm{x}(t)$ at any time $t$ is given by:


