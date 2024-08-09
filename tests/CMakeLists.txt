cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Vectoradd LANGUAGES CXX)
include(CTest)

execute_process(COMMAND rocminfo COMMAND grep gfx COMMAND sed "s/Name://" COMMAND head -1 COMMAND sed "s/ //g" OUTPUT_VARIABLE GFXMODEL)
#string(STRIP GFXMODEL GFXMODEL)
#message(STATUS "GFXMODEL ${GFXMODEL}")

set(LIBOMPTARGET_INFO_SUPPORT 1)
set(SLURM_BATCH_WAIT 1)

# Module and software install tests
#add_test(NAME Module_Test_Omniperf_1.1.0-PR1 COMMAND ../module_test_omniperf_1.1.0-PR1.sh )
#set_property(TEST Module_Test_Omniperf_1.1.0-PR1 PROPERTY PASS_REGULAR_EXPRESSION "Omniperf version: 1.1.0-PR1")

#add_test(NAME Module_Test_Omniperf_1.0.10 COMMAND ../module_test_omniperf_1.0.10.sh )
#set_property(TEST Module_Test_Omniperf_1.0.10 PROPERTY PASS_REGULAR_EXPRESSION "Omniperf version: 1.0.10")

#add_test(NAME Module_Test_Omnitrace COMMAND ../module_test_omnitrace.sh )
#set_property(TEST Module_Test_Omnitrace PROPERTY PASS_REGULAR_EXPRESSION "omnitrace-instrument v1.11.2")
#
#add_test(NAME Module_Test_AOMP_AMDClang COMMAND ../module_test_aomp_amdclang.sh )
#set_property(TEST Module_Test_AOMP_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "AOMP_STANDALONE_19.0.0 clang version 19.0.0_AOMP_STANDALONE_19.0.0")

# HIPStdPar C++
add_test(NAME HIPStdPar_Saxpy_Foreach COMMAND ../hipstdpar_saxpy_foreach.sh )
set_property(TEST HIPStdPar_Saxpy_Foreach PROPERTY PASS_REGULAR_EXPRESSION "Finished Run")

add_test(NAME HIPStdPar_Saxpy_Transform COMMAND ../hipstdpar_saxpy_transform.sh )
set_property(TEST HIPStdPar_Saxpy_Transform PROPERTY PASS_REGULAR_EXPRESSION "Finished Run")

add_test(NAME HIPStdPar_Saxpy_Transform_Reduce COMMAND ../hipstdpar_saxpy_transform_reduce.sh )
set_property(TEST HIPStdPar_Saxpy_Transform_Reduce PROPERTY PASS_REGULAR_EXPRESSION "Finished Run: Result 5120.000000")

add_test(NAME HIPStdPar_TSP COMMAND ../hipstdpar_tsp.sh )
set_property(TEST HIPStdPar_TSP PROPERTY PASS_REGULAR_EXPRESSION "Solution route is")

add_test(NAME HIPStdPar_ShallowWater_Orig COMMAND ../hipstdpar_shallowwater_orig.sh )
set_property(TEST HIPStdPar_ShallowWater_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME HIPStdPar_ShallowWater_Ver1 COMMAND ../hipstdpar_shallowwater_ver1.sh )
set_property(TEST HIPStdPar_ShallowWater_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

add_test(NAME HIPStdPar_ShallowWater_Ver2 COMMAND ../hipstdpar_shallowwater_ver2.sh )
set_property(TEST HIPStdPar_ShallowWater_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

#add_test(NAME HIPStdPar_ShallowWater_StdPar COMMAND ../hipstdpar_shallowwater_stdpar.sh )
#set_property(TEST HIPStdPar_ShallowWater_StdPar PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")

# Unified Shared Memory Examples (MI300A programming)
add_test(NAME USM_Vector_Add COMMAND ../usm_vector_add.sh )
set_property(TEST USM_Vector_Add PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME USM_Vector_Auto_Zero_Copy COMMAND ../usm_vector_auto_zero_copy.sh )
set_property(TEST USM_Vector_Auto_Zero_Copy PROPERTY PASS_REGULAR_EXPRESSION "Success")

# HIP Optimizations Examples
add_test(NAME HIP_Optimizations_daxpy_1 COMMAND ../hip_optimizations_daxpy_1.sh )
set_property(TEST HIP_Optimizations_daxpy_1 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")

add_test(NAME HIP_Optimizations_daxpy_2 COMMAND ../hip_optimizations_daxpy_2.sh )
set_property(TEST HIP_Optimizations_daxpy_2 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")

add_test(NAME HIP_Optimizations_daxpy_3 COMMAND ../hip_optimizations_daxpy_3.sh )
set_property(TEST HIP_Optimizations_daxpy_3 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")

add_test(NAME HIP_Optimizations_daxpy_4 COMMAND ../hip_optimizations_daxpy_4.sh )
set_property(TEST HIP_Optimizations_daxpy_4 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")

add_test(NAME HIP_Optimizations_daxpy_5 COMMAND ../hip_optimizations_daxpy_5.sh )
set_property(TEST HIP_Optimizations_daxpy_5 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")

# HIP-OpenMP Interoperability Examples
add_test(NAME HIP_OpenMP_Interoperability_daxpy COMMAND ../hip_openmp_interoperability_daxpy.sh )
set_property(TEST HIP_OpenMP_Interoperability_daxpy PROPERTY PASS_REGULAR_EXPRESSION "PASS")

# Programming Model Examples

add_test(NAME Programming_Model_CPU COMMAND ../programming_model_cpu.sh )
set_property(TEST Programming_Model_CPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

add_test(NAME Programming_Model_GPU COMMAND ../programming_model_gpu.sh )
set_property(TEST Programming_Model_GPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")

add_test(NAME Programming_Model_Managed_Memory COMMAND ../programming_model_managed_memory.sh )
set_property(TEST Programming_Model_Managed_Memory PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")

add_test(NAME Programming_Model_APU COMMAND ../programming_model_apu.sh )
set_property(TEST Programming_Model_APU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")

add_test(NAME Programming_Model_OpenMP COMMAND ../programming_model_openmp.sh )
set_property(TEST Programming_Model_OpenMP PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

add_test(NAME Programming_Model_Raja COMMAND ../programming_model_raja.sh )
set_property(TEST Programming_Model_Raja PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

add_test(NAME Programming_Model_Kokkos COMMAND ../programming_model_kokkos.sh )
set_property(TEST Programming_Model_Kokkos PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")

# OpenMP Examples

add_test(NAME OpenMP_Saxpy_C_AMDClang COMMAND ../openmp_saxpy_c_amdclang.sh )
set_property(TEST OpenMP_Saxpy_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_AMDGCC COMMAND ../openmp_saxpy_c_amdgcc.sh )
set_property(TEST OpenMP_Saxpy_C_AMDGCC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

add_test(NAME OpenMP_Saxpy_C_AOMP COMMAND ../openmp_saxpy_c_aomp.sh )
set_property(TEST OpenMP_Saxpy_C_AOMP PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

#add_test(NAME OpenMP_Saxpy_C_CLACC COMMAND ../openmp_saxpy_c_clacc.sh )
#set_property(TEST OpenMP_Saxpy_C_CLACC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

#add_test(NAME OpenMP_Saxpy_C_OG COMMAND ../openmp_saxpy_c_og.sh )
#set_property(TEST OpenMP_Saxpy_C_OG PROPERTY PASS_REGULAR_EXPRESSION "Last Value")

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

# Real-World OpenMP Language Constructs

add_test(NAME OpenMP_Language_Constructs_Fortran_Reduction_Scalar COMMAND ../openmp_language_constructs_fortran_reduction_scalar.sh )
set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Scalar PROPERTY PASS_REGULAR_EXPRESSION "ce1=     1000.000000000000      ce2=     1000.000000000000")

add_test(NAME OpenMP_Language_Constructs_Fortran_Reduction_Array COMMAND ../openmp_language_constructs_fortran_reduction_array.sh )
set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Array PROPERTY PASS_REGULAR_EXPRESSION "ce1=     1000.000000000000      ce2=     1000.000000000000")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Complex_Saxpy_AMDflang COMMAND ../openmp_language_constructs_fortran_complex_saxpy_amdflang.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_AMDflang PROPERTY PASS_REGULAR_EXPRESSION "0.00000000,2.00000000")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Complex_Saxpy_GFortran COMMAND ../openmp_language_constructs_fortran_complex_saxpy_gfortran.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_GFortran PROPERTY PASS_REGULAR_EXPRESSION "0.00000000,2.00000000")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Teams_EndLoop COMMAND ../openmp_language_constructs_fortran_teams_endloop.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Teams_EndLoop PROPERTY PASS_REGULAR_EXPRESSION "Success")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Derived_Type COMMAND ../openmp_language_constructs_fortran_dtype_derived_type.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Derived_Type PROPERTY PASS_REGULAR_EXPRESSION "10")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Mapper COMMAND ../openmp_language_constructs_fortran_dtype_mapper.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Mapper PROPERTY PASS_REGULAR_EXPRESSION "10")

add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Pointer COMMAND ../openmp_language_constructs_fortran_dtype_pointer.sh )
set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Pointer PROPERTY PASS_REGULAR_EXPRESSION "10")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Scalar_Members COMMAND ../openmp_language_constructs_fortran_dtype_scalar_members.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Scalar_Members PROPERTY PASS_REGULAR_EXPRESSION "10")

add_test(NAME OpenMP_Language_Constructs_C_Reduction_Scalar COMMAND ../openmp_language_constructs_c_reduction_scalar.sh )
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Scalar PROPERTY PASS_REGULAR_EXPRESSION "ce1 = 1000.000000 ce2 1000.000000")

add_test(NAME OpenMP_Language_Constructs_C_Reduction_Array COMMAND ../openmp_language_constructs_c_reduction_array.sh )
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Array PROPERTY PASS_REGULAR_EXPRESSION 'ce[0] = 1000.000000 ce[1] = 1000.000000')

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine COMMAND ../openmp_language_constructs_c_device_routine.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 1000.000000")

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine_wGlobalData COMMAND ../openmp_language_constructs_c_device_routine_wglobaldata.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wGlobalData PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 5500.000000")

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine_wDynGlobalData COMMAND ../openmp_language_constructs_c_device_routine_wdynglobaldata.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wDynGlobalData PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 5500.000000")

# HIPFort Examples

# known failure
#add_test(NAME HIPFort_GEMM_Global COMMAND ../hipfort_gemm_global.sh )
#set_property(TEST HIPFort_GEMM_Global PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME HIPFort_GEMM_Local COMMAND ../hipfort_gemm_local.sh )
set_property(TEST HIPFort_GEMM_Local PROPERTY PASS_REGULAR_EXPRESSION "Success")

# known failure
#add_test(NAME HIPFort_GEMM_Global_SD COMMAND ../hipfort_gemm_global_sd.sh )
#set_property(TEST HIPFort_GEMM_Global_SD PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME HIPFort_GEMM_Local_SD COMMAND ../hipfort_gemm_local_sd.sh )
set_property(TEST HIPFort_GEMM_Local_SD PROPERTY PASS_REGULAR_EXPRESSION "Success")


# Advanced OpenMP Examples

add_test(NAME AdvancedOpenMP_Memory1 COMMAND ../advancedopenmp_memory1.sh )
set_property(TEST AdvancedOpenMP_Memory1 PROPERTY PASS_REGULAR_EXPRESSION "Entering OpenMP kernel at mem1.cc")

add_test(NAME AdvancedOpenMP_Memory2 COMMAND ../advancedopenmp_memory2.sh )
set_property(TEST AdvancedOpenMP_Memory2 PROPERTY PASS_REGULAR_EXPRESSION "Exiting OpenMP data region at mem2.cc;Exiting OpenMP data region with end_mapper at mem2.cc")

# GPU-Aware MPI tests

add_test(NAME GPUAwareOpenMPI_pt2pt COMMAND ../gpuawarempi_pt2pt.sh )
set_property(TEST GPUAwareOpenMPI_pt2pt PROPERTY PASS_REGULAR_EXPRESSION "Run successful: received buffer has the right value")
set_tests_properties(GPUAwareOpenMPI_pt2pt PROPERTIES SKIP_RETURN_CODE "Skip")

add_test(NAME OpenMPI_Hello_World_Run COMMAND ../openmpi_hello_world_run.sh )
set_property(TEST OpenMPI_Hello_World_Run PROPERTY PASS_REGULAR_EXPRESSION "processor")

add_test(NAME OpenMPI_Check_ROCm_Support COMMAND ../openmpi_check_rocm_support.sh )
set_property(TEST OpenMPI_Check_ROCm_Support PROPERTY PASS_REGULAR_EXPRESSION "MPI extensions: affinity, cuda, ftmpi, rocm")

#add_test(NAME Mvapich2_Hello_World_C_Compile COMMAND ../mvapich2_hello_world_c_compile.sh )
#set_property(TEST Mvapich2_Hello_World_C_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_CXX_Compile COMMAND ../mvapich2_hello_world_cxx_compile.sh )
#set_property(TEST Mvapich2_Hello_World_CXX_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_C++_Compile COMMAND ../mvapich2_hello_world_c++_compile.sh )
#set_property(TEST Mvapich2_Hello_World_C++_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_F77_Compile COMMAND ../mvapich2_hello_world_f77_compile.sh )
#set_property(TEST Mvapich2_Hello_World_F77_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_F90_Compile COMMAND ../mvapich2_hello_world_f90_compile.sh )
#set_property(TEST Mvapich2_Hello_World_F90_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_Fort_Compile COMMAND ../mvapich2_hello_world_fort_compile.sh )
#set_property(TEST Mvapich2_Hello_World_Fort_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#
#add_test(NAME Mvapich2_Hello_World_Run COMMAND ../mvapich2_hello_world_run.sh )
#set_property(TEST Mvapich2_Hello_World_Run PROPERTY PASS_REGULAR_EXPRESSION "processor")

add_test(NAME GPUAwareMPI_OSUBenchmark_OpenMPI COMMAND ../gpuawarempi_osubenchmark_openmpi.sh )
set_property(TEST GPUAwareMPI_OSUBenchmark_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "8388608")

#add_test(NAME GPUAwareMPI_OSUBenchmark_Mvapich2 COMMAND ../gpuawarempi_osubenchmark_mvapich2.sh )
#set_property(TEST GPUAwareMPI_OSUBenchmark_Mvapich2 PROPERTY PASS_REGULAR_EXPRESSION "1048576")

add_test(NAME MPI_Ghost_Exchange_Orig_OpenMPI COMMAND ../mpi_ghost_exchange_orig.sh )
set_property(TEST MPI_Ghost_Exchange_Orig_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")

add_test(NAME MPI_Ghost_Exchange_Orig_Affinity_OpenMPI COMMAND ../mpi_ghost_exchange_orig_affinity.sh )
set_property(TEST MPI_Ghost_Exchange_Orig_Affinity_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")

#   add_test(NAME MPI_Ghost_Exchange_Ver1 COMMAND ../mpi_ghost_exchange_ver1.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver2 COMMAND ../mpi_ghost_exchange_ver2.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "perfetto")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver3 COMMAND ../mpi_ghost_exchange_ver3.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver4 COMMAND ../mpi_ghost_exchange_ver4.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver5 COMMAND ../mpi_ghost_exchange_ver5.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver5 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver6 COMMAND ../mpi_ghost_exchange_ver6.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver6 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver7 COMMAND ../mpi_ghost_exchange_ver7.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver7 PROPERTY PASS_REGULAR_EXPRESSION "perfetto")

# HIP Examples

add_test(NAME HIP_Vectoradd_Makefile COMMAND ../hip_vectoradd_makefile.sh )
set_property(TEST HIP_Vectoradd_Makefile PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME HIP_Vectoradd_CMakeLists COMMAND ../hip_vectoradd_cmakelists.sh )
set_property(TEST HIP_Vectoradd_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

#   add_test(NAME HIP_Vectoradd_Makefile_Batch COMMAND ../hip_vectoradd_makefile_batch.sh )
#   set_property(TEST HIP_Vectoradd_Makefile_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

#   add_test(NAME HIP_Vectoradd_CMakeLists_Batch COMMAND ../hip_vectoradd_cmakelists_batch.sh )
#   set_property(TEST HIP_Vectoradd_CMakeLists_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

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

add_test(NAME HIP_Babel_Stream COMMAND ../run_BabelStream.sh )
set_property(TEST HIP_Babel_Stream PROPERTY PASS_REGULAR_EXPRESSION "Function")

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

add_test(NAME Test_Kokkos_Install COMMAND ../test_kokkos_install.sh )
set_property( TEST Test_Kokkos_Install PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")

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

add_test(NAME NodeMemModel_Managed_Memory_HIP2_XNACK_ON COMMAND ../nodememmodel_managed_memory_hip2_xnack_on.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_ON PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

# This test should fail since it initializes the memory on the CPU and it doesn't exist.
# Why does it pass on MI200 series but fails on gfx1030?
add_test(NAME NodeMemModel_Managed_Memory_HIP3 COMMAND ../nodememmodel_managed_memory_hip3.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP3 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME NodeMemModel_Managed_Memory_HIP4 COMMAND ../nodememmodel_managed_memory_hip4.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP4 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

#Node Memory Model OpenMP Atomics

add_test(NAME NodeMemModel_OpenMP_Atomics1 COMMAND ../nodememmodel_openmp_atomics1.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics1 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_Off COMMAND ../nodememmodel_openmp_atomics2_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_On COMMAND ../nodememmodel_openmp_atomics2_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_Off COMMAND ../nodememmodel_openmp_atomics3_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_On COMMAND ../nodememmodel_openmp_atomics3_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_Off COMMAND ../nodememmodel_openmp_atomics4_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")

add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_On COMMAND ../nodememmodel_openmp_atomics4_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

#add_test(NAME NodeMemModel_OpenMP_Atomics5 COMMAND ../nodememmodel_openmp_atomics5.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics5 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

#add_test(NAME NodeMemModel_OpenMP_Atomics6 COMMAND ../nodememmodel_openmp_atomics6.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics6 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

add_test(NAME NodeMemModel_OpenMP_Atomics7 COMMAND ../nodememmodel_openmp_atomics7.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics7 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

#add_test(NAME NodeMemModel_OpenMP_Atomics8 COMMAND ../nodememmodel_openmp_atomics8.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics8 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")

#add_test(NAME NodeMemModel_OpenMP_Atomics9 COMMAND ../nodememmodel_openmp_atomics9.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics9 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

#add_test(NAME NodeMemModel_OpenMP_Atomics10 COMMAND ../nodememmodel_openmp_atomics10.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics10 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")

# Affinity tests

add_test(NAME Affinity_MPI4_OpenMP2_OpenMPI COMMAND ../affinity_mpi4_openmp2.sh )
set_property(TEST Affinity_MPI4_OpenMP2_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MPI 003 - OMP 001")

add_test(NAME Affinity_MPI2_OpenMP2_OpenMPI COMMAND ../affinity_mpi2_openmp2.sh )
set_property(TEST Affinity_MPI2_OpenMP2_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MPI 001 - OMP 001")

# bound onto socket 1 for MI300A
#add_test(NAME Affinity_MPI4_OpenMP2_Bind_OpenMPI COMMAND ../affinity_mpi4_openmp2_bind.sh )
#set_property(TEST Affinity_MPI4_OpenMP2_Bind_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MCW rank 3 bound to socket 0")

# Rocprof tests

add_test(NAME Rocprof_Mini_Nbody_Results COMMAND ../rocprof_mini_nbody_results.sh )
set_property(TEST Rocprof_Mini_Nbody_Results PROPERTY PASS_REGULAR_EXPRESSION "10")

add_test(NAME Rocprof_Mini_Nbody_Stats COMMAND ../rocprof_mini_nbody_stats.sh )
set_property(TEST Rocprof_Mini_Nbody_Stats PROPERTY PASS_REGULAR_EXPRESSION "bodyForce")

add_test(NAME Rocprof_Mini_Nbody_Basenames COMMAND ../rocprof_mini_nbody_basenames.sh )
set_property(TEST Rocprof_Mini_Nbody_Basenames PROPERTY PASS_REGULAR_EXPRESSION "bodyForce")

add_test(NAME Rocprof_Mini_Nbody_HIP_Trace COMMAND ../rocprof_mini_nbody_hip_trace.sh )
set_property(TEST Rocprof_Mini_Nbody_HIP_Trace PROPERTY PASS_REGULAR_EXPRESSION "hipLaunchKernel")

add_test(NAME Rocprof_Mini_Nbody_HSA_Trace COMMAND ../rocprof_mini_nbody_hsa_trace.sh )
set_property(TEST Rocprof_Mini_Nbody_HSA_Trace PROPERTY PASS_REGULAR_EXPRESSION "hsa_system_get_info")

# PyTorch tests

add_test(NAME Pytorch_Check_Import COMMAND ../pytorch_check_import.sh )
set_property(TEST Pytorch_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME Pytorch_Torchvision_Check_Import COMMAND ../torchvision_check_import.sh )
set_property(TEST Pytorch_Torchvision_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME Pytorch_Torchaudio_Check_Import COMMAND ../torchaudio_check_import.sh )
set_property(TEST Pytorch_Torchaudio_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME Pytorch_Check_Gpu_Access COMMAND ../pytorch_check_gpu_access.sh )
set_property(TEST Pytorch_Check_Gpu_Access PROPERTY PASS_REGULAR_EXPRESSION "True")

add_test(NAME Pytorch_Check_Gpu_Name COMMAND ../pytorch_check_gpu_name.sh )
set_property(TEST Pytorch_Check_Gpu_Name PROPERTY PASS_REGULAR_EXPRESSION "AMD")

add_test(NAME Pytorch_Check_ROCm_Dependency COMMAND ../pytorch_check_rocm_dependency.sh )
set_property(TEST Pytorch_Check_ROCm_Dependency PROPERTY PASS_REGULAR_EXPRESSION "Success")

#add_test(NAME Pytorch_Unit_Tests COMMAND ../pytorch_unit_tests.py )
#set_property(TEST Pytorch_Unit_Tests PROPERTY PASS_REGULAR_EXPRESSION "47559 passed")

add_test(NAME Pytorch_Mnist COMMAND ../pytorch_mnist.sh )
set_property(TEST Pytorch_Mnist PROPERTY PASS_REGULAR_EXPRESSION "99%")

# Python MPI tests

add_test(NAME MPI4PY_Check_Import COMMAND ../mpi4py_check_import.sh )
set_property(TEST MPI4PY_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME MPI4PY_Check_Import_MPI COMMAND ../mpi4py_check_import_mpi.sh )
set_property(TEST MPI4PY_Check_Import_MPI PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME MPI4PY_Cupy COMMAND ../mpi4py_cupy.sh )
set_property(TEST MPI4PY_Cupy PROPERTY PASS_REGULAR_EXPRESSION "Success")

# Cupy tests

add_test(NAME Cupy_Check_Import COMMAND ../cupy_check_import.sh )
set_property(TEST Cupy_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")

add_test(NAME Cupy_Install_Tests COMMAND ../cupy_install_tests.sh )
set_property(TEST Cupy_Install_Tests PROPERTY PASS_REGULAR_EXPRESSION "11 passed")

# Omnitrace tests

add_test(NAME Omnitrace_Avail_Check COMMAND ../omnitrace-avail_check.sh )
set_property(TEST Omnitrace_Avail_Check PROPERTY PASS_REGULAR_EXPRESSION "Outputting text configuration file ")

add_test(NAME Omnitrace_Instrument_Check COMMAND ../omnitrace-instrument_check.sh )
set_property(TEST Omnitrace_Instrument_Check PROPERTY PASS_REGULAR_EXPRESSION "The instrumented executable image is stored in ")

add_test(NAME Omnitrace_Run_Check COMMAND ../omnitrace-run_check.sh )
set_property(TEST Omnitrace_Run_Check PROPERTY PASS_REGULAR_EXPRESSION "Tracing session 1 ended, total sessions:0")

add_test(NAME Omnitrace_Stream_Overlap COMMAND ../omnitrace_stream_overlap.sh )
set_property(TEST Omnitrace_Stream_Overlap PROPERTY PASS_REGULAR_EXPRESSION "proto")

# Omniperf Tests

add_test(NAME Omniperf_Roofline_Check COMMAND ../omniperf_roofline_check.sh )
set_property(TEST Omniperf_Roofline_Check PROPERTY PASS_REGULAR_EXPRESSION "rooflines_PDF")

add_test(NAME Omniperf_Profile_Check COMMAND ../omniperf_profile_check.sh )
set_property(TEST Omniperf_Profile_Check PROPERTY PASS_REGULAR_EXPRESSION "is generating")

add_test(NAME Omniperf_Analyze_Check COMMAND ../omniperf_analyze_check.sh )
set_property(TEST Omniperf_Analyze_Check PROPERTY PASS_REGULAR_EXPRESSION "deriving Omniperf metrics...")
