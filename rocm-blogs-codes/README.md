<!---
Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->
# AMD lab notes

Computational and Data science have emerged as powerful modes of scientific inquiry and engineering design. 
Often referred to as the "third" and "fourth" pillars of the scientific method, they are interdisciplinary 
fields where computer models and simulations of physical, biological, or data-driven processes are used to 
probe, predict, and analyze complex systems of interest.
All of this necessitates the use of more computational 
power and resources to keep up with increasing scientific and industrial demands. In order to fully 
utilize emerging hardware designed to tackle these challenges, the development of robust software 
for high-performance computing (HPC) and Machine Learning (ML) applications is now more crucial than ever. This challenge 
is made even greater as hardware trends continue to achieve massive parallelism through GPU 
acceleration, which requires the adoption of sophisticated heterogenous programming environments 
and carefully tuned application code.

In this "AMD lab notes" blog series, we share the lessons learned from tuning a wide range of scientific applications,
libraries, and frameworks for AMD GPUs.
Our goal with these lab notes is to provide readers with the following:

- AMD GPU implementations of computational science algorithms such as PDE discretizations, 
linear algebra, solvers, and more
- AMD GPU programming tutorials showcasing optimizations
- Instructions for leveraging ML frameworks, data science tools, post-processing, and visualization on AMD GPUs
- Best practices for porting and optimizing HPC and ML applications targeting AMD GPUs
- Guidance on using libraries and tools from the ROCm™ software stack

Most of our lab notes contain accompanying code examples that readers are encouraged to experiment with.
The intention is to provide content that targets domain experts and 
computational/data scientists alike. While our optimization strategies may be specific to a particular 
application, we believe that the content can serve as loose guidelines and an effective starting point 
for getting the best experience out of AMD hardware. We primarily focus on AMD Instinct™ GPUs,
but we expect users of other AMD graphics cards to also benefit from the strategies outlined in these notes.

The repository containing all lab notes and associated code examples can be found at 
[https://github.com/AMD/amd-lab-notes](https://github.com/AMD/amd-lab-notes).
We hope that our pedagogical examples will inspire readers to accelerate their application code even further.

If you have any questions or comments, please reach out to us on GitHub [Discussions](https://github.com/amd/amd-lab-notes/discussions)
