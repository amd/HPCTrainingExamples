# Solve a Linear System of ODEs with ROCm Libraries

This simple example will show how to use some of the available ROCm libraries to solve a simple system of linear ordinary differential equations (ODEs).
These systems are solved to describe, for instance, the evolution of dynamical systems, wherthe velocity of the system at a given time depends linearly on the position. The term linear here is intended in the mathematical sense.

## Problem Definition

Consider the initial value problem:

$$
\dot{\mathbf{x}} = A \mathbf{x}

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
\cos(t) - \sin(t) \\
\sin(t) \cos(t)
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
