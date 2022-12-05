Originally from HIP-Examples/openmp-helloworld in
   https://github.com/ROCm-Developer-Tools/HIP-Examples

# Simple OpenMP hello world example written directly to the HIP interface.

## Requirements
* Installed ROCm 3.9 or newer. See  [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

## Windows Requirements
* Set HIP_DIR to the HIP installation location.
* libamdhip64.dll and amd_comgr.dll must be in PATH or in System32.
* Install MS Visual Studio 2019 for C++ development with Optional C++ Clang tools for Windows.
* Ensure libomp.dll from MSVC C++ Clang tools is in PATH (by default, location is C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\Llvm\x64\bin).
* Modify the CMakeLists.txt of this project to the corresponding libomp.lib location.

## How to run this code:

### Using Make on Linux:
* To build and run: `make`.
* To clean the environment: `make clean`.


### Using CMake on Linux:
* To build: `mkdir -p build; cd build; cmake ..; make`
* To run the test: `./test_openmp_helloworld`
* To clean the build environment: `make clean`

### Using CMake on Windows:
* CMake Command: `cmake -G Ninja -DCMAKE_C_COMPILER=<HIP_DIR>/bin/clang.exe -DCMAKE_CXX_COMPILER=<HIP_DIR>/bin/clang++.exe`
* To build: `ninja`
* To run the test: `./test_openmp_helloworld`

**Note:** You may override `AMDGPU_TARGETS` in the HIP config file by modifying the CMakeLists.txt.

## Expected Results:
```
info: running on device Device 66a3
Hello World... from OMP thread = 0
Hello World... from OMP thread = 15
Hello World... from OMP thread = 3
Hello World... from OMP thread = 13
Hello World... from OMP thread = 11
Hello World... from OMP thread = 8
Hello World... from OMP thread = 4
Hello World... from OMP thread = 1
Hello World... from OMP thread = 10
Hello World... from OMP thread = 9
Hello World... from OMP thread = 7
Hello World... from OMP thread = 12
Hello World... from OMP thread = 6
Hello World... from OMP thread = 14
Hello World... from OMP thread = 5
Hello World... from OMP thread = 2
Hello World... from HIP thread = 0
Hello World... from HIP thread = 2
Hello World... from HIP thread = 5
Hello World... from HIP thread = 14
Hello World... from HIP thread = 6
Hello World... from HIP thread = 12
Hello World... from HIP thread = 7
Hello World... from HIP thread = 9
Hello World... from HIP thread = 1
Hello World... from HIP thread = 11
Hello World... from HIP thread = 10
Hello World... from HIP thread = 4
Hello World... from HIP thread = 8
Hello World... from HIP thread = 13
Hello World... from HIP thread = 15
Hello World... from HIP thread = 3
Device Results:
  A_d[0] = 0
  A_d[1] = 1
  A_d[2] = 2
  A_d[3] = 3
  A_d[4] = 4
  A_d[5] = 5
  A_d[6] = 6
  A_d[7] = 7
  A_d[8] = 8
  A_d[9] = 9
  A_d[10] = 10
  A_d[11] = 11
  A_d[12] = 12
  A_d[13] = 13
  A_d[14] = 14
  A_d[15] = 15
PASSED!
```

**Note:** HIP thread's printf may not display on builds with printf support disabled.
