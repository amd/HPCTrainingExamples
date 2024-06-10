These examples are for use in AMD training and testing. They are released under the MIT license as
described in the License.txt file in the main directory of the Examples. Any copies or use of these
examples shall include the license and copyright notice.

These examples are tested on the Thera system, an internal AMD testing 
system. Modules and environment files are for that system. They give
an idea of what environment should be set up for other systems.

To run on Thera. Will run all four compilers with both OpenMP
and OpenACC and report whether the compilation is successful.
   ./runall.sh

Example code
A simple vector add example

The compilers can be specified with the CC and FC variables. 
The rocminfo should be in the path (${ROCM_PATH}/bin) or
the GPU model can be specified in the ROCM_GPU environment variable.

To set up environment on Thera, source the *_env file

To compile and run in separate steps, first set up your environment
and then:

make
./vecadd
