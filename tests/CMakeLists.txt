cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Vectoradd LANGUAGES CXX)
include(CTest)

execute_process(COMMAND rocminfo COMMAND grep gfx COMMAND sed "s/Name://" COMMAND head -1 COMMAND sed "s/ //g" OUTPUT_VARIABLE GFXMODEL)
#string(STRIP GFXMODEL GFXMODEL)
#message(STATUS "GFXMODEL ${GFXMODEL}")

execute_process(COMMAND rocminfo COMMAND grep GPU\&"Device Type" COMMAND wc -l OUTPUT_VARIABLE GPUCOUNT)
string(STRIP GPUCOUNT GPUCOUNT)
message(STATUS "GPUCOUNT ${GPUCOUNT}")

execute_process(COMMAND rocminfo COMMAND grep xnack COMMAND wc -l OUTPUT_VARIABLE XNACK_COUNT)
#string(STRIP XNACK_COUNT XNACK_COUNT)
#message(STATUS "XNACK_COUNT ${XNACK_COUNT}")

if(${GFXMODEL} MATCHES "gfx1030")
   set(LIBOMPTARGET_INFO_SUPPORT 0)
   set(SLURM_BATCH_WAIT 0)
else ()
   set(LIBOMPTARGET_INFO_SUPPORT 1)
   set(SLURM_BATCH_WAIT 1)
endif()

# Programming Model Examples

add_test(NAME Programming_Model_CPU COMMAND ../programming_model_cpu.sh )
set_property(TEST Programming_Model_CPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

add_test(NAME Programming_Model_GPU COMMAND ../programming_model_gpu.sh )
set_property(TEST Programming_Model_GPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME Programming_Model_Managed_Memory COMMAND ../programming_model_managed_memory.sh )
   set_property(TEST Programming_Model_Managed_Memory PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

   add_test(NAME Programming_Model_APU COMMAND ../programming_model_apu.sh )
   set_property(TEST Programming_Model_APU PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")
endif ()

add_test(NAME Programming_Model_OpenMP COMMAND ../programming_model_openmp.sh )
set_property(TEST Programming_Model_OpenMP PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME Programming_Model_Raja COMMAND ../programming_model_raja.sh )
   set_property(TEST Programming_Model_Raja PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

   add_test(NAME Programming_Model_Kokkos COMMAND ../programming_model_kokkos.sh )
   set_property(TEST Programming_Model_Kokkos PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")
endif ()

# OpenMP Examples

add_test(NAME OpenMP_Saxpy_C_AMDClang COMMAND ../openmp_saxpy_c_amdclang.sh )
set_property(TEST OpenMP_Saxpy_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_AMDGCC COMMAND ../openmp_saxpy_c_amdgcc.sh )
set_property(TEST OpenMP_Saxpy_C_AMDGCC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_AOMP COMMAND ../openmp_saxpy_c_aomp.sh )
set_property(TEST OpenMP_Saxpy_C_AOMP PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_CLACC COMMAND ../openmp_saxpy_c_clacc.sh )
set_property(TEST OpenMP_Saxpy_C_CLACC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_OG COMMAND ../openmp_saxpy_c_og.sh )
set_property(TEST OpenMP_Saxpy_C_OG PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

## System clang not build with AMD offloading
#add_test(NAME OpenMP_Saxpy_C_Clang COMMAND ../openmp_saxpy_c_clang.sh )
#set_property(TEST OpenMP_Saxpy_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_GCC COMMAND ../openmp_saxpy_c_gcc.sh )
set_property(TEST OpenMP_Saxpy_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Freduce_F_AMDFlang COMMAND ../openmp_freduce_f_amdflang.sh )
set_property(TEST OpenMP_Freduce_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "1010")

## No flang-15 installed in /usr/bin
## Linux system flang not built with AMD offloading
#add_test(NAME OpenMP_Freduce_F_Flang COMMAND ../openmp_freduce_f_flang.sh )
#set_property(TEST OpenMP_Freduce_F_Flang PROPERTY PASS_REGULAR_EXPRESSION "1010")

add_test(NAME OpenMP_Freduce_F_GFortran COMMAND ../openmp_freduce_f_gfortran.sh )
set_property(TEST OpenMP_Freduce_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "1010")

add_test(NAME OpenMP_Vecadd_F_AMDFlang COMMAND ../openmp_vecadd_f_amdflang.sh )
set_property(TEST OpenMP_Vecadd_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "Final result:   1.000000")

add_test(NAME OpenMP_Vecadd_F_GFortran COMMAND ../openmp_vecadd_f_gfortran.sh )
set_property(TEST OpenMP_Vecadd_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "Final result:   1.000000")

# OpenMP Intro Examples

add_test(NAME OpenMP_Intro_Saxpy_CPU COMMAND ../openmp_intro_saxpy_cpu.sh )
set_property(TEST OpenMP_Intro_Saxpy_CPU PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy1 COMMAND ../openmp_intro_saxpy1.sh )
set_property(TEST OpenMP_Intro_Saxpy1 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy2 COMMAND ../openmp_intro_saxpy2.sh )
set_property(TEST OpenMP_Intro_Saxpy2 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy1f COMMAND ../openmp_intro_saxpy1f.sh )
set_property(TEST OpenMP_Intro_Saxpy1f PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy3 COMMAND ../openmp_intro_saxpy3.sh )
set_property(TEST OpenMP_Intro_Saxpy3 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy4 COMMAND ../openmp_intro_saxpy4.sh )
set_property(TEST OpenMP_Intro_Saxpy4 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy5 COMMAND ../openmp_intro_saxpy5.sh )
set_property(TEST OpenMP_Intro_Saxpy5 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

# Failing with segmentation fault
#add_test(NAME OpenMP_Intro_Saxpy6 COMMAND ../openmp_intro_saxpy6.sh )
#set_property(TEST OpenMP_Intro_Saxpy6 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy7 COMMAND ../openmp_intro_saxpy7.sh )
set_property(TEST OpenMP_Intro_Saxpy7 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Saxpy2f COMMAND ../openmp_intro_saxpy2f.sh )
set_property(TEST OpenMP_Intro_Saxpy2f PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")

add_test(NAME OpenMP_Intro_Target_Data_Structured COMMAND ../openmp_intro_target_data_structured.sh )
set_property(TEST OpenMP_Intro_Target_Data_Structured PROPERTY PASS_REGULAR_EXPRESSION "completed successfully")

add_test(NAME OpenMP_Intro_Target_Data_Unstructured COMMAND ../openmp_intro_target_data_unstructured.sh )
set_property(TEST OpenMP_Intro_Target_Data_Unstructured PROPERTY PASS_REGULAR_EXPRESSION "completed successfully")

add_test(NAME OpenMP_Intro_Target_Data_Update COMMAND ../openmp_intro_target_data_update.sh )
set_property(TEST OpenMP_Intro_Target_Data_Update PROPERTY PASS_REGULAR_EXPRESSION "Target Update result is 300000.000000")

# Additional OpenMP Language Constructs

# Advanced OpenMP Examples

if(${LIBOMPTARGET_INFO_SUPPORT} MATCHES "1")
   add_test(NAME AdvancedOpenMP_Memory1 COMMAND ../advancedopenmp_memory1.sh )
   set_property(TEST AdvancedOpenMP_Memory1 PROPERTY PASS_REGULAR_EXPRESSION "Entering OpenMP kernel at mem1.cc")

   add_test(NAME AdvancedOpenMP_Memory2 COMMAND ../advancedopenmp_memory2.sh )
   set_property(TEST AdvancedOpenMP_Memory2 PROPERTY PASS_REGULAR_EXPRESSION "Exiting OpenMP data region at mem2.cc")
endif()

# HIP Examples

add_test(NAME HIP_Vectoradd_Makefile COMMAND ../hip_vectoradd_makefile.sh )
set_property(TEST HIP_Vectoradd_Makefile PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Vectoradd_CMakeLists COMMAND ../hip_vectoradd_cmakelists.sh )
set_property(TEST HIP_Vectoradd_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

if(${SLURM_BATCH_WAIT} MATCHES "1")
   add_test(NAME HIP_Vectoradd_Makefile_Batch COMMAND ../hip_vectoradd_makefile_batch.sh )
   set_property(TEST HIP_Vectoradd_Makefile_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

   add_test(NAME HIP_Vectoradd_CMakeLists_Batch COMMAND ../hip_vectoradd_cmakelists_batch.sh )
   set_property(TEST HIP_Vectoradd_CMakeLists_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
endif()

add_test(NAME HIP_Vectoradd_Makefile_Batch_Direct COMMAND ../hip_vectoradd_makefile_batch_direct.sh )
set_property(TEST HIP_Vectoradd_Makefile_Batch_Direct PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Vectoradd_CMakeLists_Batch_Direct COMMAND ../hip_vectoradd_cmakelists_batch_direct.sh )
set_property(TEST HIP_Vectoradd_CMakeLists_Batch_Direct PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Stream_Makefile COMMAND ../hip_stream_makefile.sh )
set_property(TEST HIP_Stream_Makefile PROPERTY PASS_REGULAR_EXPRESSION "Triad:")

add_test(NAME HIP_Stream_CMakeLists COMMAND ../hip_stream_cmakelists.sh )
set_property(TEST HIP_Stream_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "Triad:")

add_test(NAME HIP_Saxpy_Makefile COMMAND ../hip_saxpy_makefile.sh )
set_property(TEST HIP_Saxpy_Makefile PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Saxpy_CMakeLists COMMAND ../hip_saxpy_cmakelists.sh )
set_property(TEST HIP_Saxpy_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Jacobi_CMakeLists COMMAND ../hip_jacobi_cmakelists.sh )
set_property(TEST HIP_Jacobi_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "Measured FLOPS")

# HIPIFY Examples

add_test(NAME Hipify_Nbody COMMAND ../hipify_nbody.sh )
set_property(TEST Hipify_Nbody PROPERTY PASS_REGULAR_EXPRESSION "3000")

# OpenACC Examples

#add_test(NAME OpenACC_Saxpy_C_AMDClang COMMAND ../openacc_saxpy_c_amdclang.sh )
#set_property(TEST OpenACC_Saxpy_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

#add_test(NAME OpenACC_Saxpy_C_Clang COMMAND ../openacc_saxpy_c_clang.sh )
#set_property(TEST OpenACC_Saxpy_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

# Error with libgomp
#add_test(NAME OpenACC_Saxpy_C_GCC COMMAND ../openacc_saxpy_c_gcc.sh )
#set_property(TEST OpenACC_Saxpy_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "Final result")

#add_test(NAME OpenACC_Vecadd_C_AMDClang COMMAND ../openacc_vecadd_c_amdclang.sh )
#set_property(TEST OpenACC_Vecadd_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

#add_test(NAME OpenACC_Vecadd_C_Clang COMMAND ../openacc_vecadd_c_clang.sh )
#set_property(TEST OpenACC_Vecadd_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

# math library missing?
#add_test(NAME OpenACC_Vecadd_C_GCC COMMAND ../openacc_vecadd_c_gcc.sh )
#set_property(TEST OpenACC_Vecadd_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "Final result")

#add_test(NAME OpenACC_Vecadd_F_AMDFlang COMMAND ../openacc_vecadd_f_amdflang.sh )
#set_property(TEST OpenACC_Vecadd_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

#add_test(NAME OpenACC_Vecadd_F_Flang COMMAND ../openacc_vecadd_f_flang.sh )
#set_property(TEST OpenACC_Vecadd_F_Flang PROPERTY PASS_REGULAR_EXPRESSION "Final result")

# failing
#add_test(NAME OpenACC_Vecadd_F_GFortran COMMAND ../openacc_vecadd_f_gfortran.sh )
#set_property(TEST OpenACC_Vecadd_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "Final result")

# Kokkos

add_test(NAME Kokkos_OpenMP_Stream_Orig COMMAND ../kokkos_openmp_stream_orig.sh )
set_property(TEST Kokkos_OpenMP_Stream_Orig PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_OpenMP_Stream_Ver1 COMMAND ../kokkos_openmp_stream_ver1.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_OpenMP_Stream_Ver2 COMMAND ../kokkos_openmp_stream_ver2.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_OpenMP_Stream_Ver3 COMMAND ../kokkos_openmp_stream_ver3.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_OpenMP_Stream_Ver4 COMMAND ../kokkos_openmp_stream_ver4.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_HIP_Stream_Orig COMMAND ../kokkos_hip_stream_orig.sh )
set_property(TEST Kokkos_HIP_Stream_Orig PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_HIP_Stream_Ver1 COMMAND ../kokkos_hip_stream_ver1.sh )
set_property(TEST Kokkos_HIP_Stream_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_HIP_Stream_Ver2 COMMAND ../kokkos_hip_stream_ver2.sh )
set_property(TEST Kokkos_HIP_Stream_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_HIP_Stream_Ver3 COMMAND ../kokkos_hip_stream_ver3.sh )
set_property(TEST Kokkos_HIP_Stream_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_HIP_Stream_Ver4 COMMAND ../kokkos_hip_stream_ver4.sh )
set_property(TEST Kokkos_HIP_Stream_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

add_test(NAME Kokkos_ShallowWater_Serial_Orig COMMAND ../kokkos_shallow_water_serial_orig.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_Serial_Par1 COMMAND ../kokkos_shallow_water_serial_par1.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_Serial_Par2 COMMAND ../kokkos_shallow_water_serial_par2.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_Serial_Par3 COMMAND ../kokkos_shallow_water_serial_par3.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_Serial_Par4 COMMAND ../kokkos_shallow_water_serial_par4.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_Serial_Par5 COMMAND ../kokkos_shallow_water_serial_par5.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Orig COMMAND ../kokkos_shallow_water_openmp_orig.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par1 COMMAND ../kokkos_shallow_water_openmp_par1.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par2 COMMAND ../kokkos_shallow_water_openmp_par2.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par3 COMMAND ../kokkos_shallow_water_openmp_par3.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par4 COMMAND ../kokkos_shallow_water_openmp_par4.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par5 COMMAND ../kokkos_shallow_water_openmp_par5.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_HIP_Orig COMMAND ../kokkos_shallow_water_hip_orig.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_HIP_Par1 COMMAND ../kokkos_shallow_water_hip_par1.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par1 PROPERTY PASS_REGULAR_EXPRESSION "attempt to access inaccessible memory space")

add_test(NAME Kokkos_ShallowWater_HIP_Par2 COMMAND ../kokkos_shallow_water_hip_par2.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par2 PROPERTY PASS_REGULAR_EXPRESSION "attempt to access inaccessible memory space")

add_test(NAME Kokkos_ShallowWater_HIP_Par3 COMMAND ../kokkos_shallow_water_hip_par3.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_HIP_Par4 COMMAND ../kokkos_shallow_water_hip_par4.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME Kokkos_ShallowWater_HIP_Par5 COMMAND ../kokkos_shallow_water_hip_par5.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

#Node Memory Model Managed Memory

add_test(NAME NodeMemModel_Managed_Memory_HIP_Orig COMMAND ../nodememmodel_managed_memory_hip_orig.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP_Orig PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME NodeMemModel_Managed_Memory_HIP1 COMMAND ../nodememmodel_managed_memory_hip1.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP1 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME NodeMemModel_Managed_Memory_HIP2_XNACK_OFF COMMAND ../nodememmodel_managed_memory_hip2_xnack_off.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_OFF PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_Managed_Memory_HIP2_XNACK_ON COMMAND ../nodememmodel_managed_memory_hip2_xnack_on.sh )
   set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_ON PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
endif ()

# This test should fail since it initializes the memory on the CPU and it doesn't exist.
# Why does it pass on MI200 series but fails on gfx1030?
if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_Managed_Memory_HIP3 COMMAND ../nodememmodel_managed_memory_hip3.sh )
   set_property(TEST NodeMemModel_Managed_Memory_HIP3 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
endif()

add_test(NAME NodeMemModel_Managed_Memory_HIP4 COMMAND ../nodememmodel_managed_memory_hip4.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP4 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

#Node Memory Model OpenMP Atomics

add_test(NAME NodeMemModel_OpenMP_Atomics1 COMMAND ../nodememmodel_openmp_atomics1.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics1 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_Off COMMAND ../nodememmodel_openmp_atomics2_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_On COMMAND ../nodememmodel_openmp_atomics2_xnack_on.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
endif()

add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_Off COMMAND ../nodememmodel_openmp_atomics3_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_On COMMAND ../nodememmodel_openmp_atomics3_xnack_on.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
endif()

add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_Off COMMAND ../nodememmodel_openmp_atomics4_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_On COMMAND ../nodememmodel_openmp_atomics4_xnack_on.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
endif()

if(${XNACK_COUNT} GREATER_EQUAL "1")
   add_test(NAME NodeMemModel_OpenMP_Atomics5 COMMAND ../nodememmodel_openmp_atomics5.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics5 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

   add_test(NAME NodeMemModel_OpenMP_Atomics6 COMMAND ../nodememmodel_openmp_atomics6.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics6 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

   add_test(NAME NodeMemModel_OpenMP_Atomics7 COMMAND ../nodememmodel_openmp_atomics7.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics7 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

   add_test(NAME NodeMemModel_OpenMP_Atomics8 COMMAND ../nodememmodel_openmp_atomics8.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics8 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

   add_test(NAME NodeMemModel_OpenMP_Atomics9 COMMAND ../nodememmodel_openmp_atomics9.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics9 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

   add_test(NAME NodeMemModel_OpenMP_Atomics10 COMMAND ../nodememmodel_openmp_atomics10.sh )
   set_property(TEST NodeMemModel_OpenMP_Atomics10 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
endif()

# GPU-Aware MPI tests

if (${GPUCOUNT} GREATER_EQUAL "2")
   add_test(NAME GPUAwareMPI_pt2pt COMMAND ../gpuawarempi_pt2pt.sh )
   set_property(TEST GPUAwareMPI_pt2pt PROPERTY PASS_REGULAR_EXPRESSION "Run successful: received buffer has the right value")
endif()

add_test(NAME GPUAwareMPI_OSUBenchmark COMMAND ../gpuawarempi_osubenchmark.sh )
set_property(TEST GPUAwareMPI_OSUBenchmark PROPERTY PASS_REGULAR_EXPRESSION "8388608")

# Affinity tests

add_test(NAME Affinity_MPI4_OpenMP2 COMMAND ../affinity_mpi4_openmp2.sh )
set_property(TEST Affinity_MPI4_OpenMP2 PROPERTY PASS_REGULAR_EXPRESSION "MPI 003 - OMP 001")

add_test(NAME Affinity_MPI2_OpenMP2 COMMAND ../affinity_mpi2_openmp2.sh )
set_property(TEST Affinity_MPI2_OpenMP2 PROPERTY PASS_REGULAR_EXPRESSION "MPI 001 - OMP 001")

add_test(NAME Affinity_MPI4_OpenMP2_Bind COMMAND ../affinity_mpi4_openmp2_bind.sh )
set_property(TEST Affinity_MPI4_OpenMP2_Bind PROPERTY PASS_REGULAR_EXPRESSION "MCW rank 3 bound to socket 0")
