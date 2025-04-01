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
#set_property(TEST Module_Test_Omniperf_1.1.0-PR1 PROPERTY PASS_REGULAR_EXPRESSION "Rocprof-compute version: 1.1.0-PR1")
#set_property(TEST Module_Test_Omniperf_1.1.0-PR1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME Module_Test_Omniperf_1.0.10 COMMAND ../module_test_omniperf_1.0.10.sh )
#set_property(TEST Module_Test_Omniperf_1.0.10 PROPERTY PASS_REGULAR_EXPRESSION "Rocprof-compute version: 1.0.10")
#set_property(TEST Module_Test_Omniperf_1.0.10 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME Module_Test_Omnitrace COMMAND ../module_test_omnitrace.sh )
#set_property(TEST Module_Test_Omnitrace PROPERTY PASS_REGULAR_EXPRESSION "omnitrace-instrument v1.11.2")
#set_property(TEST Module_Test_Omnitrace PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME Module_Test_AOMP_AMDClang COMMAND ../module_test_aomp_amdclang.sh )
#set_property(TEST Module_Test_AOMP_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "AOMP_STANDALONE_19.0.0 clang version 19.0.0_AOMP_STANDALONE_19.0.0")
#set_property(TEST Module_Test_AOMP_AMDClang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIP basic functionalities
add_test(NAME HIP_Query_Device COMMAND ../hip_query_device.sh )
set_property(TEST HIP_Query_Device PROPERTY PASS_REGULAR_EXPRESSION "SUCCESS")
set_property(TEST HIP_Query_Device PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIPStdPar C++
add_test(NAME HIPStdPar_Saxpy_Foreach COMMAND ../hipstdpar_saxpy_foreach.sh )
set_property(TEST HIPStdPar_Saxpy_Foreach PROPERTY PASS_REGULAR_EXPRESSION "Finished Run")
set_property(TEST HIPStdPar_Saxpy_Foreach PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_Saxpy_Transform COMMAND ../hipstdpar_saxpy_transform.sh )
set_property(TEST HIPStdPar_Saxpy_Transform PROPERTY PASS_REGULAR_EXPRESSION "Finished Run")
set_property(TEST HIPStdPar_Saxpy_Transform PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_Saxpy_Transform_Reduce COMMAND ../hipstdpar_saxpy_transform_reduce.sh )
set_property(TEST HIPStdPar_Saxpy_Transform_Reduce PROPERTY PASS_REGULAR_EXPRESSION "Finished Run: Result 5120.000000")
set_property(TEST HIPStdPar_Saxpy_Transform_Reduce PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_TSP COMMAND ../hipstdpar_tsp.sh )
set_property(TEST HIPStdPar_TSP PROPERTY PASS_REGULAR_EXPRESSION "Solution route is")
set_property(TEST HIPStdPar_TSP PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_ShallowWater_Orig COMMAND ../hipstdpar_shallowwater_orig.sh )
set_property(TEST HIPStdPar_ShallowWater_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST HIPStdPar_ShallowWater_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_ShallowWater_Ver1 COMMAND ../hipstdpar_shallowwater_ver1.sh )
set_property(TEST HIPStdPar_ShallowWater_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST HIPStdPar_ShallowWater_Ver1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPStdPar_ShallowWater_Ver2 COMMAND ../hipstdpar_shallowwater_ver2.sh )
set_property(TEST HIPStdPar_ShallowWater_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST HIPStdPar_ShallowWater_Ver2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME HIPStdPar_ShallowWater_StdPar COMMAND ../hipstdpar_shallowwater_stdpar.sh )
#set_property(TEST HIPStdPar_ShallowWater_StdPar PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
#set_property(TEST HIPStdPar_ShallowWater_StdPar PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Unified Shared Memory Examples (MI300A programming)
add_test(NAME USM_Vector_Add COMMAND ../usm_vector_add.sh )
set_property(TEST USM_Vector_Add PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST USM_Vector_Add PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME USM_Vector_Auto_Zero_Copy COMMAND ../usm_vector_auto_zero_copy.sh )
set_property(TEST USM_Vector_Auto_Zero_Copy PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST USM_Vector_Auto_Zero_Copy PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIP Optimizations Examples
add_test(NAME HIP_Optimizations_daxpy_1 COMMAND ../hip_optimizations_daxpy_1.sh )
set_property(TEST HIP_Optimizations_daxpy_1 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")
set_property(TEST HIP_Optimizations_daxpy_1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Optimizations_daxpy_2 COMMAND ../hip_optimizations_daxpy_2.sh )
set_property(TEST HIP_Optimizations_daxpy_2 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")
set_property(TEST HIP_Optimizations_daxpy_2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Optimizations_daxpy_3 COMMAND ../hip_optimizations_daxpy_3.sh )
set_property(TEST HIP_Optimizations_daxpy_3 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")
set_property(TEST HIP_Optimizations_daxpy_3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Optimizations_daxpy_4 COMMAND ../hip_optimizations_daxpy_4.sh )
set_property(TEST HIP_Optimizations_daxpy_4 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")
set_property(TEST HIP_Optimizations_daxpy_4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Optimizations_daxpy_5 COMMAND ../hip_optimizations_daxpy_5.sh )
set_property(TEST HIP_Optimizations_daxpy_5 PROPERTY PASS_REGULAR_EXPRESSION "PASSED")
set_property(TEST HIP_Optimizations_daxpy_5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIP-OpenMP Interoperability Examples
add_test(NAME HIP_OpenMP_Interoperability_daxpy COMMAND ../hip_openmp_interoperability_daxpy.sh )
set_property(TEST HIP_OpenMP_Interoperability_daxpy PROPERTY PASS_REGULAR_EXPRESSION "PASS")
set_property(TEST HIP_OpenMP_Interoperability_daxpy PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME HIP_OpenMP_Interoperability_saxpy_offload COMMAND ../hip_openmp_interoperability_saxpy_offload.sh )
set_property(TEST HIP_OpenMP_Interoperability_saxpy_offload PROPERTY PASS_REGULAR_EXPRESSION "PASS")
set_property(TEST HIP_OpenMP_Interoperability_saxpy_offload PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME HIP_OpenMP_Interoperability_saxpy_hip COMMAND ../hip_openmp_interoperability_saxpy_hip.sh )
set_property(TEST HIP_OpenMP_Interoperability_saxpy_hip PROPERTY PASS_REGULAR_EXPRESSION "PASS")
set_property(TEST HIP_OpenMP_Interoperability_saxpy_hip PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME HIP_OpenMP_Interoperability_saxpy_APU COMMAND ../hip_openmp_interoperability_saxpy_APU.sh )
set_property(TEST HIP_OpenMP_Interoperability_saxpy_APU PROPERTY PASS_REGULAR_EXPRESSION "PASS")
set_property(TEST HIP_OpenMP_Interoperability_saxpy_APU PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

# Programming Model Examples

add_test(NAME Programming_Model_CPU COMMAND ../programming_model_cpu.sh )
set_property(TEST Programming_Model_CPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST Programming_Model_CPU PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Programming_Model_GPU COMMAND ../programming_model_gpu.sh )
set_property(TEST Programming_Model_GPU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST Programming_Model_GPU PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Programming_Model_Managed_Memory COMMAND ../programming_model_managed_memory.sh )
set_property(TEST Programming_Model_Managed_Memory PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST Programming_Model_Managed_Memory PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME Programming_Model_APU COMMAND ../programming_model_apu.sh )
set_property(TEST Programming_Model_APU PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST Programming_Model_APU PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME Programming_Model_OpenMP COMMAND ../programming_model_openmp.sh )
set_property(TEST Programming_Model_OpenMP PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST Programming_Model_OpenMP PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME Programming_Model_Raja COMMAND ../programming_model_raja.sh )
set_property(TEST Programming_Model_Raja PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")
set_property(TEST Programming_Model_Raja PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME Programming_Model_Kokkos COMMAND ../programming_model_kokkos.sh )
set_property(TEST Programming_Model_Kokkos PROPERTY PASS_REGULAR_EXPRESSION "Result is 2000.000000")
set_property(TEST Programming_Model_Kokkos PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")


# OpenMP Build System Examples

add_test(NAME OpenMP_C_Build_Make COMMAND ../openmp_C_build_make.sh )
set_property(TEST OpenMP_C_Build_Make PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_C_Build_Make PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMP_CXX_Build_Make COMMAND ../openmp_CXX_build_make.sh )
set_property(TEST OpenMP_CXX_Build_Make PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_CXX_Build_Make PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMP_Fortran_Build_Make COMMAND ../openmp_Fortran_build_make.sh )
set_property(TEST OpenMP_Fortran_Build_Make PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_Fortran_Build_Make PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMP_C_Build_CMake COMMAND ../openmp_C_build_cmake.sh )
set_property(TEST OpenMP_C_Build_CMake PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_C_Build_CMake PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMP_CXX_Build_CMake COMMAND ../openmp_CXX_build_cmake.sh )
set_property(TEST OpenMP_CXX_Build_CMake PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_CXX_Build_CMake PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMP_Fortran_Build_CMake COMMAND ../openmp_Fortran_build_cmake.sh )
set_property(TEST OpenMP_Fortran_Build_CMake PROPERTY PASS_REGULAR_EXPRESSION "Result is 200000.000000")
set_property(TEST OpenMP_Fortran_Build_CMake PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

# OpenMP Examples

add_test(NAME OpenMP_Saxpy_C_AMDClang COMMAND ../openmp_saxpy_c_amdclang.sh )
set_property(TEST OpenMP_Saxpy_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "4.000000")
set_property(TEST OpenMP_Saxpy_C_AMDClang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Saxpy_C_AMDGCC COMMAND ../openmp_saxpy_c_amdgcc.sh )
#set_property(TEST OpenMP_Saxpy_C_AMDGCC PROPERTY PASS_REGULAR_EXPRESSION "4.000000")
#set_property(TEST OpenMP_Saxpy_C_AMDGCC PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Saxpy_C_AOMP COMMAND ../openmp_saxpy_c_aomp.sh )
set_property(TEST OpenMP_Saxpy_C_AOMP PROPERTY PASS_REGULAR_EXPRESSION "4.000000")
set_property(TEST OpenMP_Saxpy_C_AOMP PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Saxpy_C_CLACC COMMAND ../openmp_saxpy_c_clacc.sh )
#set_property(TEST OpenMP_Saxpy_C_CLACC PROPERTY PASS_REGULAR_EXPRESSION "Last Value")
#set_property(TEST OpenMP_Saxpy_C_CLACC PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Saxpy_C_OG COMMAND ../openmp_saxpy_c_og.sh )
#set_property(TEST OpenMP_Saxpy_C_OG PROPERTY PASS_REGULAR_EXPRESSION "Last Value")
#set_property(TEST OpenMP_Saxpy_C_OG PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

## System clang not build with AMD offloading
#add_test(NAME OpenMP_Saxpy_C_Clang COMMAND ../openmp_saxpy_c_clang.sh )
#set_property(TEST OpenMP_Saxpy_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Last Value")
#set_property(TEST OpenMP_Saxpy_C_Clang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Saxpy_C_GCC COMMAND ../openmp_saxpy_c_gcc.sh )
set_property(TEST OpenMP_Saxpy_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "4.000000")
set_property(TEST OpenMP_Saxpy_C_GCC PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Freduce_F_AMDFlang COMMAND ../openmp_freduce_f_amdflang.sh )
set_property(TEST OpenMP_Freduce_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "1010")
set_property(TEST OpenMP_Freduce_F_AMDFlang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

## No flang-15 installed in /usr/bin
## Linux system flang not built with AMD offloading
#add_test(NAME OpenMP_Freduce_F_Flang COMMAND ../openmp_freduce_f_flang.sh )
#set_property(TEST OpenMP_Freduce_F_Flang PROPERTY PASS_REGULAR_EXPRESSION "1010")
#set_property(TEST OpenMP_Freduce_F_Flang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Freduce_F_GFortran COMMAND ../openmp_freduce_f_gfortran.sh )
set_property(TEST OpenMP_Freduce_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "1010")
set_property(TEST OpenMP_Freduce_F_GFortran PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Vecadd_F_AMDFlang COMMAND ../openmp_vecadd_f_amdflang.sh )
set_property(TEST OpenMP_Vecadd_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "Final result:   1.000000")
set_property(TEST OpenMP_Vecadd_F_AMDFlang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Vecadd_F_GFortran COMMAND ../openmp_vecadd_f_gfortran.sh )
set_property(TEST OpenMP_Vecadd_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "Final result:   1.000000")
set_property(TEST OpenMP_Vecadd_F_GFortran PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# OpenMP Intro Examples

add_test(NAME OpenMP_Intro_Saxpy_CPU COMMAND ../openmp_intro_saxpy_cpu.sh )
set_property(TEST OpenMP_Intro_Saxpy_CPU PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy_CPU PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy1 COMMAND ../openmp_intro_saxpy1.sh )
set_property(TEST OpenMP_Intro_Saxpy1 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy2 COMMAND ../openmp_intro_saxpy2.sh )
set_property(TEST OpenMP_Intro_Saxpy2 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Intro_Saxpy1f COMMAND ../openmp_intro_saxpy1f.sh )
#set_property(TEST OpenMP_Intro_Saxpy1f PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
#set_property(TEST OpenMP_Intro_Saxpy1f PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy3 COMMAND ../openmp_intro_saxpy3.sh )
set_property(TEST OpenMP_Intro_Saxpy3 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy4 COMMAND ../openmp_intro_saxpy4.sh )
set_property(TEST OpenMP_Intro_Saxpy4 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy5 COMMAND ../openmp_intro_saxpy5.sh )
set_property(TEST OpenMP_Intro_Saxpy5 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Failing with segmentation fault
#add_test(NAME OpenMP_Intro_Saxpy6 COMMAND ../openmp_intro_saxpy6.sh )
#set_property(TEST OpenMP_Intro_Saxpy6 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
#set_property(TEST OpenMP_Intro_Saxpy6 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Saxpy7 COMMAND ../openmp_intro_saxpy7.sh )
set_property(TEST OpenMP_Intro_Saxpy7 PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
set_property(TEST OpenMP_Intro_Saxpy7 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Intro_Saxpy2f COMMAND ../openmp_intro_saxpy2f.sh )
#set_property(TEST OpenMP_Intro_Saxpy2f PROPERTY PASS_REGULAR_EXPRESSION "Time of kernel:")
#set_property(TEST OpenMP_Intro_Saxpy2f PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Target_Data_Structured COMMAND ../openmp_intro_target_data_structured.sh )
set_property(TEST OpenMP_Intro_Target_Data_Structured PROPERTY PASS_REGULAR_EXPRESSION "completed successfully")
set_property(TEST OpenMP_Intro_Target_Data_Structured PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Target_Data_Unstructured COMMAND ../openmp_intro_target_data_unstructured.sh )
set_property(TEST OpenMP_Intro_Target_Data_Unstructured PROPERTY PASS_REGULAR_EXPRESSION "completed successfully")
set_property(TEST OpenMP_Intro_Target_Data_Unstructured PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Intro_Target_Data_Update COMMAND ../openmp_intro_target_data_update.sh )
set_property(TEST OpenMP_Intro_Target_Data_Update PROPERTY PASS_REGULAR_EXPRESSION "Target Update result is 300000.000000")
set_property(TEST OpenMP_Intro_Target_Data_Update PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Real-World OpenMP Language Constructs

add_test(NAME OpenMP_Language_Constructs_Fortran_Reduction_Scalar COMMAND ../openmp_language_constructs_fortran_reduction_scalar.sh )
set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Scalar PROPERTY PASS_REGULAR_EXPRESSION "ce1=     1000.000000000000      ce2=     1000.000000000000")
set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Scalar PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenMP_Language_Constructs_Fortran_Reduction_Array COMMAND ../openmp_language_constructs_fortran_reduction_array.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Array PROPERTY PASS_REGULAR_EXPRESSION "ce1=     1000.000000000000      ce2=     1000.000000000000")
#set_property(TEST OpenMP_Language_Constructs_Fortran_Reduction_Array PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Complex_Saxpy_AMDflang COMMAND ../openmp_language_constructs_fortran_complex_saxpy_amdflang.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_AMDflang PROPERTY PASS_REGULAR_EXPRESSION "0.00000000,2.00000000")
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_AMDflang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Complex_Saxpy_GFortran COMMAND ../openmp_language_constructs_fortran_complex_saxpy_gfortran.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_GFortran PROPERTY PASS_REGULAR_EXPRESSION "0.00000000,2.00000000")
#set_property(TEST OpenMP_Language_Constructs_Fortran_Complex_Saxpy_GFortran PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_Teams_EndLoop COMMAND ../openmp_language_constructs_fortran_teams_endloop.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_Teams_EndLoop PROPERTY PASS_REGULAR_EXPRESSION "Success")
#set_property(TEST OpenMP_Language_Constructs_Fortran_Teams_EndLoop PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Derived_Type COMMAND ../openmp_language_constructs_fortran_dtype_derived_type.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Derived_Type PROPERTY PASS_REGULAR_EXPRESSION "10")
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Derived_Type PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Mapper COMMAND ../openmp_language_constructs_fortran_dtype_mapper.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Mapper PROPERTY PASS_REGULAR_EXPRESSION "10")
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Mapper PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Pointer COMMAND ../openmp_language_constructs_fortran_dtype_pointer.sh )
set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Pointer PROPERTY PASS_REGULAR_EXPRESSION "10")
set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Pointer PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Known failure
#add_test(NAME OpenMP_Language_Constructs_Fortran_DType_Scalar_Members COMMAND ../openmp_language_constructs_fortran_dtype_scalar_members.sh )
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Scalar_Members PROPERTY PASS_REGULAR_EXPRESSION "10")
#set_property(TEST OpenMP_Language_Constructs_Fortran_DType_Scalar_Members PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_C_Reduction_Scalar COMMAND ../openmp_language_constructs_c_reduction_scalar.sh )
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Scalar PROPERTY PASS_REGULAR_EXPRESSION "ce1 = 1000.000000 ce2 1000.000000")
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Scalar PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_C_Reduction_Array COMMAND ../openmp_language_constructs_c_reduction_array.sh )
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Array PROPERTY PASS_REGULAR_EXPRESSION 'ce[0] = 1000.000000 ce[1] = 1000.000000')
set_property(TEST OpenMP_Language_Constructs_C_Reduction_Array PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine COMMAND ../openmp_language_constructs_c_device_routine.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 1000.000000")
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine_wGlobalData COMMAND ../openmp_language_constructs_c_device_routine_wglobaldata.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wGlobalData PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 5500.000000")
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wGlobalData PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_C_Device_Routine_wDynGlobalData COMMAND ../openmp_language_constructs_c_device_routine_wdynglobaldata.sh )
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wDynGlobalData PROPERTY PASS_REGULAR_EXPRESSION "Result: sum of x is 5500.000000")
set_property(TEST OpenMP_Language_Constructs_C_Device_Routine_wDynGlobalData PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_CXX_Member_Function COMMAND ../openmp_language_constructs_cxx_member_function.sh )
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function PROPERTY PASS_REGULAR_EXPRESSION "Finished calculation")
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_CXX_Member_Function_PortYourself COMMAND ../openmp_language_constructs_cxx_member_function_portyourself.sh )
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_PortYourself PROPERTY PASS_REGULAR_EXPRESSION "Finished calculation")
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_PortYourself PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_CXX_Member_Function_External COMMAND ../openmp_language_constructs_cxx_member_function_external.sh )
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_External PROPERTY PASS_REGULAR_EXPRESSION "Finished calculation")
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_External PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMP_Language_Constructs_CXX_Member_Function_External_PortYourself COMMAND ../openmp_language_constructs_cxx_member_function_external_portyourself.sh )
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_External_PortYourself PROPERTY PASS_REGULAR_EXPRESSION "Finished calculation")
set_property(TEST OpenMP_Language_Constructs_CXX_Member_Function_External_PortYourself PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIPFort Examples

add_test(NAME HIPFort_Module_Check_hipfc_2003 COMMAND ../hipfort_module_check_hipfc_2003.sh )
set_property(TEST HIPFort_Module_Check_hipfc_2003 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIPFort_Module_Check_hipfc_2003 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_Module_Check_hipfc_2008 COMMAND ../hipfort_module_check_hipfc_2008.sh )
set_property(TEST HIPFort_Module_Check_hipfc_2008 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIPFort_Module_Check_hipfc_2008 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_ROCm_Include_GEMM_Global COMMAND ../hipfort_gemm_global.sh )
set_property(TEST HIPFort_ROCm_Include_GEMM_Global PROPERTY PASS_REGULAR_EXPRESSION "2048")
set_property(TEST HIPFort_ROCm_Include_GEMM_Global PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_ROCm_Include_GEMM_Local COMMAND ../hipfort_gemm_local.sh )
set_property(TEST HIPFort_ROCm_Include_GEMM_Local PROPERTY PASS_REGULAR_EXPRESSION "2048")
set_property(TEST HIPFort_ROCm_Include_GEMM_Local PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_ROCm_Include_GEMM_Global_SD COMMAND ../hipfort_gemm_global_sd.sh )
set_property(TEST HIPFort_ROCm_Include_GEMM_Global_SD PROPERTY PASS_REGULAR_EXPRESSION "2048")
set_property(TEST HIPFort_ROCm_Include_GEMM_Global_SD PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_ROCm_Include_GEMM_Local_SD COMMAND ../hipfort_gemm_local_sd.sh )
set_property(TEST HIPFort_ROCm_Include_GEMM_Local_SD PROPERTY PASS_REGULAR_EXPRESSION "2048")
set_property(TEST HIPFort_ROCm_Include_GEMM_Local_SD PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIPFort_Module_Check_Matmult COMMAND ../hipfort_module_check_matmult.sh )
set_property(TEST HIPFort_Module_Check_Matmult PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST HIPFort_Module_Check_Matmult PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Advanced OpenMP Examples

add_test(NAME AdvancedOpenMP_Memory1 COMMAND ../advancedopenmp_memory1.sh )
set_property(TEST AdvancedOpenMP_Memory1 PROPERTY PASS_REGULAR_EXPRESSION "Entering OpenMP kernel at mem1.cc")
set_property(TEST AdvancedOpenMP_Memory1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME AdvancedOpenMP_Memory2 COMMAND ../advancedopenmp_memory2.sh )
set_property(TEST AdvancedOpenMP_Memory2 PROPERTY PASS_REGULAR_EXPRESSION "Exiting OpenMP data region at mem2.cc;Exiting OpenMP data region with end_mapper at mem2.cc")
set_property(TEST AdvancedOpenMP_Memory2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

# GPU-Aware MPI tests

add_test(NAME GPUAwareOpenMPI_pt2pt COMMAND ../gpuawarempi_pt2pt.sh )
set_property(TEST GPUAwareOpenMPI_pt2pt PROPERTY PASS_REGULAR_EXPRESSION "Run successful: received buffer has the right value")
set_property(TEST GPUAwareOpenMPI_pt2pt PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME OpenMPI_Hello_World_Run COMMAND ../openmpi_hello_world_run.sh )
set_property(TEST OpenMPI_Hello_World_Run PROPERTY PASS_REGULAR_EXPRESSION "Hello world from processor")
set_property(TEST OpenMPI_Hello_World_Run PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME OpenMPI_Check_ROCm_Support COMMAND ../openmpi_check_rocm_support.sh )
set_property(TEST OpenMPI_Check_ROCm_Support PROPERTY PASS_REGULAR_EXPRESSION "MPI extensions: affinity, cuda, ftmpi, rocm")
set_property(TEST OpenMPI_Check_ROCm_Support PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME Mvapich2_Hello_World_C_Compile COMMAND ../mvapich2_hello_world_c_compile.sh )
#set_property(TEST Mvapich2_Hello_World_C_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_C_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_CXX_Compile COMMAND ../mvapich2_hello_world_cxx_compile.sh )
#set_property(TEST Mvapich2_Hello_World_CXX_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_CXX_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_C++_Compile COMMAND ../mvapich2_hello_world_c++_compile.sh )
#set_property(TEST Mvapich2_Hello_World_C++_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_C++_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_F77_Compile COMMAND ../mvapich2_hello_world_f77_compile.sh )
#set_property(TEST Mvapich2_Hello_World_F77_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_F77_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_F90_Compile COMMAND ../mvapich2_hello_world_f90_compile.sh )
#set_property(TEST Mvapich2_Hello_World_F90_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_F90_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_Fort_Compile COMMAND ../mvapich2_hello_world_fort_compile.sh )
#set_property(TEST Mvapich2_Hello_World_Fort_Compile PROPERTY PASS_REGULAR_EXPRESSION "Executable created")
#set_property(TEST Mvapich2_Hello_World_Fort_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#add_test(NAME Mvapich2_Hello_World_Run COMMAND ../mvapich2_hello_world_run.sh )
#set_property(TEST Mvapich2_Hello_World_Run PROPERTY PASS_REGULAR_EXPRESSION "processor")
#set_property(TEST Mvapich2_Hello_World_Run_Compile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME GPUAwareMPI_OSUBenchmark_OpenMPI COMMAND ../gpuawarempi_osubenchmark_openmpi.sh )
set_property(TEST GPUAwareMPI_OSUBenchmark_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "8388608")
set_property(TEST GPUAwareMPI_OSUBenchmark_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME GPUAwareMPI_OSUBenchmark_Mvapich2 COMMAND ../gpuawarempi_osubenchmark_mvapich2.sh )
#set_property(TEST GPUAwareMPI_OSUBenchmark_Mvapich2 PROPERTY PASS_REGULAR_EXPRESSION "1048576")
#set_property(TEST GPUAwareMPI_OSUBenchmark_Mvapich2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME MPI_Ghost_Exchange_Orig_OpenMPI COMMAND ../mpi_ghost_exchange_orig.sh )
set_property(TEST MPI_Ghost_Exchange_Orig_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
set_property(TEST MPI_Ghost_Exchange_Orig_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME MPI_Ghost_Exchange_Orig_Affinity_OpenMPI COMMAND ../mpi_ghost_exchange_orig_affinity.sh )
set_property(TEST MPI_Ghost_Exchange_Orig_Affinity_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
set_property(TEST MPI_Ghost_Exchange_Orig_Affinity_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#   add_test(NAME MPI_Ghost_Exchange_Ver1 COMMAND ../mpi_ghost_exchange_ver1.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#   set_property(TEST MPI_Ghost_Exchange_Ver1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver2 COMMAND ../mpi_ghost_exchange_ver2.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "perfetto")
#   set_property(TEST MPI_Ghost_Exchange_Ver2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver3 COMMAND ../mpi_ghost_exchange_ver3.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#   set_property(TEST MPI_Ghost_Exchange_Ver3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver4 COMMAND ../mpi_ghost_exchange_ver4.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#   set_property(TEST MPI_Ghost_Exchange_Ver4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver5 COMMAND ../mpi_ghost_exchange_ver5.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver5 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#   set_property(TEST MPI_Ghost_Exchange_Ver5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver6 COMMAND ../mpi_ghost_exchange_ver6.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver6 PROPERTY PASS_REGULAR_EXPRESSION "GhostExchange_ArrayAssign Timing is stencil")
#   set_property(TEST MPI_Ghost_Exchange_Ver6 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
#
#   add_test(NAME MPI_Ghost_Exchange_Ver7 COMMAND ../mpi_ghost_exchange_ver7.sh )
#   set_property(TEST MPI_Ghost_Exchange_Ver7 PROPERTY PASS_REGULAR_EXPRESSION "perfetto")
#   set_property(TEST MPI_Ghost_Exchange_Ver7 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIP Examples

add_test(NAME HIP_Vectoradd_Makefile COMMAND ../hip_vectoradd_makefile.sh )
set_property(TEST HIP_Vectoradd_Makefile PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Vectoradd_Makefile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Vectoradd_CMakeLists COMMAND ../hip_vectoradd_cmakelists.sh )
set_property(TEST HIP_Vectoradd_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Vectoradd_CMakeLists PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#   add_test(NAME HIP_Vectoradd_Makefile_Batch COMMAND ../hip_vectoradd_makefile_batch.sh )
#   set_property(TEST HIP_Vectoradd_Makefile_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
#   set_property(TEST HIP_Vectoradd_Makefile_Batch PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#   add_test(NAME HIP_Vectoradd_CMakeLists_Batch COMMAND ../hip_vectoradd_cmakelists_batch.sh )
#   set_property(TEST HIP_Vectoradd_CMakeLists_Batch PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
#   set_property(TEST HIP_Vectoradd_CMakeLists_Batch PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Vectoradd_Makefile_Batch_Direct COMMAND ../hip_vectoradd_makefile_batch_direct.sh )
set_property(TEST HIP_Vectoradd_Makefile_Batch_Direct PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Vectoradd_Makefile_Batch_Direct PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Vectoradd_CMakeLists_Batch_Direct COMMAND ../hip_vectoradd_cmakelists_batch_direct.sh )
set_property(TEST HIP_Vectoradd_CMakeLists_Batch_Direct PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Vectoradd_CMakeLists_Batch_Direct PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Stream_Makefile COMMAND ../hip_stream_makefile.sh )
set_property(TEST HIP_Stream_Makefile PROPERTY PASS_REGULAR_EXPRESSION "Triad:")
set_property(TEST HIP_Stream_Makefile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Stream_CMakeLists COMMAND ../hip_stream_cmakelists.sh )
set_property(TEST HIP_Stream_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "Triad:")
set_property(TEST HIP_Stream_CMakeLists PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Saxpy_Makefile COMMAND ../hip_saxpy_makefile.sh )
set_property(TEST HIP_Saxpy_Makefile PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Saxpy_Makefile PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Saxpy_CMakeLists COMMAND ../hip_saxpy_cmakelists.sh )
set_property(TEST HIP_Saxpy_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST HIP_Saxpy_CMakeLists PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Jacobi_CMakeLists COMMAND ../hip_jacobi_cmakelists.sh )
set_property(TEST HIP_Jacobi_CMakeLists PROPERTY PASS_REGULAR_EXPRESSION "Measured FLOPS")
set_property(TEST HIP_Jacobi_CMakeLists PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HIP_Babel_Stream COMMAND ../run_BabelStream.sh )
set_property(TEST HIP_Babel_Stream PROPERTY PASS_REGULAR_EXPRESSION "Function")
set_property(TEST HIP_Babel_Stream PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIPIFY Examples

add_test(NAME Hipify_Nbody COMMAND ../hipify_nbody.sh )
set_property(TEST Hipify_Nbody PROPERTY PASS_REGULAR_EXPRESSION "3000")
set_property(TEST Hipify_Nbody PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HIPIFLY Examples

add_test(NAME Hipifly_Without_Module COMMAND ../hipifly_without_module.sh )
set_property(TEST Hipifly_Without_Module PROPERTY PASS_REGULAR_EXPRESSION "Validation PASSED.")
set_property(TEST Hipifly_Without_Module PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Hipifly_With_Module COMMAND ../hipifly_with_module.sh )
set_property(TEST Hipifly_With_Module PROPERTY PASS_REGULAR_EXPRESSION "Validation PASSED.")
set_property(TEST Hipifly_With_Module PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# OpenACC Examples

#add_test(NAME OpenACC_Saxpy_C_AMDClang COMMAND ../openacc_saxpy_c_amdclang.sh )
#set_property(TEST OpenACC_Saxpy_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Saxpy_C_AMDClang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenACC_Saxpy_C_Clang COMMAND ../openacc_saxpy_c_clang.sh )
#set_property(TEST OpenACC_Saxpy_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Saxpy_C_Clang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Error with libgomp
#add_test(NAME OpenACC_Saxpy_C_GCC COMMAND ../openacc_saxpy_c_gcc.sh )
#set_property(TEST OpenACC_Saxpy_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Saxpy_C_GCC PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenACC_Vecadd_C_AMDClang COMMAND ../openacc_vecadd_c_amdclang.sh )
#set_property(TEST OpenACC_Vecadd_C_AMDClang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_C_AMDClang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenACC_Vecadd_C_Clang COMMAND ../openacc_vecadd_c_clang.sh )
#set_property(TEST OpenACC_Vecadd_C_Clang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_C_Clang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# math library missing?
#add_test(NAME OpenACC_Vecadd_C_GCC COMMAND ../openacc_vecadd_c_gcc.sh )
#set_property(TEST OpenACC_Vecadd_C_GCC PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_C_GCC PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenACC_Vecadd_F_AMDFlang COMMAND ../openacc_vecadd_f_amdflang.sh )
#set_property(TEST OpenACC_Vecadd_F_AMDFlang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_F_AMDFlang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME OpenACC_Vecadd_F_Flang COMMAND ../openacc_vecadd_f_flang.sh )
#set_property(TEST OpenACC_Vecadd_F_Flang PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_F_Flang PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# failing
#add_test(NAME OpenACC_Vecadd_F_GFortran COMMAND ../openacc_vecadd_f_gfortran.sh )
#set_property(TEST OpenACC_Vecadd_F_GFortran PROPERTY PASS_REGULAR_EXPRESSION "Final result")
#set_property(TEST OpenACC_Vecadd_F_GFortran PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Kokkos

add_test(NAME Kokkos_Test_Install COMMAND ../kokkos_test_install.sh )
set_property( TEST Kokkos_Test_Install PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_Test_Install PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_OpenMP_Stream_Orig COMMAND ../kokkos_openmp_stream_orig.sh )
set_property(TEST Kokkos_OpenMP_Stream_Orig PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_OpenMP_Stream_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_OpenMP_Stream_Ver1 COMMAND ../kokkos_openmp_stream_ver1.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_OpenMP_Stream_Ver1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_OpenMP_Stream_Ver2 COMMAND ../kokkos_openmp_stream_ver2.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_OpenMP_Stream_Ver2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_OpenMP_Stream_Ver3 COMMAND ../kokkos_openmp_stream_ver3.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_OpenMP_Stream_Ver3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_OpenMP_Stream_Ver4 COMMAND ../kokkos_openmp_stream_ver4.sh )
set_property(TEST Kokkos_OpenMP_Stream_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_OpenMP_Stream_Ver4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_HIP_Stream_Orig COMMAND ../kokkos_hip_stream_orig.sh )
set_property(TEST Kokkos_HIP_Stream_Orig PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_HIP_Stream_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_HIP_Stream_Ver1 COMMAND ../kokkos_hip_stream_ver1.sh )
set_property(TEST Kokkos_HIP_Stream_Ver1 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_HIP_Stream_Ver1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_HIP_Stream_Ver2 COMMAND ../kokkos_hip_stream_ver2.sh )
set_property(TEST Kokkos_HIP_Stream_Ver2 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_HIP_Stream_Ver2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_HIP_Stream_Ver3 COMMAND ../kokkos_hip_stream_ver3.sh )
set_property(TEST Kokkos_HIP_Stream_Ver3 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_HIP_Stream_Ver3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_HIP_Stream_Ver4 COMMAND ../kokkos_hip_stream_ver4.sh )
set_property(TEST Kokkos_HIP_Stream_Ver4 PROPERTY PASS_REGULAR_EXPRESSION "Average runtime")
set_property(TEST Kokkos_HIP_Stream_Ver4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Orig COMMAND ../kokkos_shallow_water_serial_orig.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Par1 COMMAND ../kokkos_shallow_water_serial_par1.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Par1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Par2 COMMAND ../kokkos_shallow_water_serial_par2.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Par2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Par3 COMMAND ../kokkos_shallow_water_serial_par3.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Par3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Par4 COMMAND ../kokkos_shallow_water_serial_par4.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Par4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_Serial_Par5 COMMAND ../kokkos_shallow_water_serial_par5.sh )
set_property(TEST Kokkos_ShallowWater_Serial_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_Serial_Par5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Orig COMMAND ../kokkos_shallow_water_openmp_orig.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par1 COMMAND ../kokkos_shallow_water_openmp_par1.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par1 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Par1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par2 COMMAND ../kokkos_shallow_water_openmp_par2.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par2 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Par2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par3 COMMAND ../kokkos_shallow_water_openmp_par3.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Par3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par4 COMMAND ../kokkos_shallow_water_openmp_par4.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Par4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_OpenMP_Par5 COMMAND ../kokkos_shallow_water_openmp_par5.sh )
set_property(TEST Kokkos_ShallowWater_OpenMP_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_OpenMP_Par5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Orig COMMAND ../kokkos_shallow_water_hip_orig.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Orig PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_HIP_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Par1 COMMAND ../kokkos_shallow_water_hip_par1.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par1 PROPERTY PASS_REGULAR_EXPRESSION "attempt to access inaccessible memory space")
set_property(TEST Kokkos_ShallowWater_HIP_Par1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Par2 COMMAND ../kokkos_shallow_water_hip_par2.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par2 PROPERTY PASS_REGULAR_EXPRESSION "attempt to access inaccessible memory space")
set_property(TEST Kokkos_ShallowWater_HIP_Par2 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Par3 COMMAND ../kokkos_shallow_water_hip_par3.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par3 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_HIP_Par3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Par4 COMMAND ../kokkos_shallow_water_hip_par4.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par4 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_HIP_Par4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Kokkos_ShallowWater_HIP_Par5 COMMAND ../kokkos_shallow_water_hip_par5.sh )
set_property(TEST Kokkos_ShallowWater_HIP_Par5 PROPERTY PASS_REGULAR_EXPRESSION "Flow finished")
set_property(TEST Kokkos_ShallowWater_HIP_Par5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#Node Memory Model Managed Memory

add_test(NAME NodeMemModel_Managed_Memory_HIP_Orig COMMAND ../nodememmodel_managed_memory_hip_orig.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP_Orig PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST NodeMemModel_Managed_Memory_HIP_Orig PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_Managed_Memory_HIP1 COMMAND ../nodememmodel_managed_memory_hip1.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP1 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST NodeMemModel_Managed_Memory_HIP1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_Managed_Memory_HIP2_XNACK_OFF COMMAND ../nodememmodel_managed_memory_hip2_xnack_off.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_OFF PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_OFF PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_Managed_Memory_HIP2_XNACK_ON COMMAND ../nodememmodel_managed_memory_hip2_xnack_on.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_ON PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST NodeMemModel_Managed_Memory_HIP2_XNACK_ON PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

# This test should fail since it initializes the memory on the CPU and it doesn't exist.
# Why does it pass on MI200 series but fails on gfx1030?
add_test(NAME NodeMemModel_Managed_Memory_HIP3 COMMAND ../nodememmodel_managed_memory_hip3.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP3 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST NodeMemModel_Managed_Memory_HIP3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME NodeMemModel_Managed_Memory_HIP4 COMMAND ../nodememmodel_managed_memory_hip4.sh )
set_property(TEST NodeMemModel_Managed_Memory_HIP4 PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")
set_property(TEST NodeMemModel_Managed_Memory_HIP4 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#Node Memory Model OpenMP Atomics

add_test(NAME NodeMemModel_OpenMP_Atomics1 COMMAND ../nodememmodel_openmp_atomics1.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics1 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
set_property(TEST NodeMemModel_OpenMP_Atomics1 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_Off COMMAND ../nodememmodel_openmp_atomics2_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_Off PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_OpenMP_Atomics2_XNACK_On COMMAND ../nodememmodel_openmp_atomics2_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_On PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_Off COMMAND ../nodememmodel_openmp_atomics3_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_Off PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_OpenMP_Atomics3_XNACK_On COMMAND ../nodememmodel_openmp_atomics3_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics3_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
set_property(TEST NodeMemModel_OpenMP_Atomics2_XNACK_On PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_Off COMMAND ../nodememmodel_openmp_atomics4_xnack_off.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_Off PROPERTY PASS_REGULAR_EXPRESSION "Memory access fault")
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_Off PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME NodeMemModel_OpenMP_Atomics4_XNACK_On COMMAND ../nodememmodel_openmp_atomics4_xnack_on.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_On PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
set_property(TEST NodeMemModel_OpenMP_Atomics4_XNACK_On PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

#add_test(NAME NodeMemModel_OpenMP_Atomics5 COMMAND ../nodememmodel_openmp_atomics5.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics5 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")
#set_property(TEST NodeMemModel_OpenMP_Atomics5 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

#add_test(NAME NodeMemModel_OpenMP_Atomics6 COMMAND ../nodememmodel_openmp_atomics6.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics6 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")
#set_property(TEST NodeMemModel_OpenMP_Atomics6 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME NodeMemModel_OpenMP_Atomics7 COMMAND ../nodememmodel_openmp_atomics7.sh )
set_property(TEST NodeMemModel_OpenMP_Atomics7 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
set_property(TEST NodeMemModel_OpenMP_Atomics7 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

#add_test(NAME NodeMemModel_OpenMP_Atomics8 COMMAND ../nodememmodel_openmp_atomics8.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics8 PROPERTY PASS_REGULAR_EXPRESSION "FAILED")
#set_property(TEST NodeMemModel_OpenMP_Atomics8 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

#add_test(NAME NodeMemModel_OpenMP_Atomics9 COMMAND ../nodememmodel_openmp_atomics9.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics9 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
#set_property(TEST NodeMemModel_OpenMP_Atomics9 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

#add_test(NAME NodeMemModel_OpenMP_Atomics10 COMMAND ../nodememmodel_openmp_atomics10.sh )
#set_property(TEST NodeMemModel_OpenMP_Atomics10 PROPERTY PASS_REGULAR_EXPRESSION "Test PASSED")
#set_property(TEST NodeMemModel_OpenMP_Atomics10 PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

# Affinity tests

add_test(NAME Affinity_MPI4_OpenMP2_OpenMPI COMMAND ../affinity_mpi4_openmp2.sh )
set_property(TEST Affinity_MPI4_OpenMP2_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MPI 003 - OMP 001")
set_property(TEST Affinity_MPI4_OpenMP2_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Affinity_MPI2_OpenMP2_OpenMPI COMMAND ../affinity_mpi2_openmp2.sh )
set_property(TEST Affinity_MPI2_OpenMP2_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MPI 001 - OMP 001")
set_property(TEST Affinity_MPI2_OpenMP2_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# bound onto socket 1 for MI300A
#add_test(NAME Affinity_MPI4_OpenMP2_Bind_OpenMPI COMMAND ../affinity_mpi4_openmp2_bind.sh )
#set_property(TEST Affinity_MPI4_OpenMP2_Bind_OpenMPI PROPERTY PASS_REGULAR_EXPRESSION "MCW rank 3 bound to socket 0")
#set_property(TEST Affinity_MPI4_OpenMP2_Bind_OpenMPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# RocprofV3 tests

add_test(NAME RocprofV3_Mini_Nbody_Kernel_Stats COMMAND ../rocprofv3_mini_nbody_stats_sys_trace.sh )
set_property(TEST RocprofV3_Mini_Nbody_Kernel_Stats PROPERTY PASS_REGULAR_EXPRESSION "bodyForce")
set_property(TEST RocprofV3_Mini_Nbody_Kernel_Stats PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME RocprofV3_Mini_Nbody_Marker_Trace COMMAND ../rocprofv3_mini_nbody_stats_sys_trace.sh )
set_property(TEST RocprofV3_Mini_Nbody_Marker_Trace PROPERTY PASS_REGULAR_EXPRESSION "randomizeBodies_marker")
set_property(TEST RocprofV3_Mini_Nbody_Marker_Trace PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME RocprofV3_Mini_Nbody_HIP_Trace COMMAND ../rocprofv3_mini_nbody_stats_sys_trace.sh )
set_property(TEST RocprofV3_Mini_Nbody_HIP_Trace PROPERTY PASS_REGULAR_EXPRESSION "hipLaunchKernel")
set_property(TEST RocprofV3_Mini_Nbody_HIP_Trace PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME RocprofV3_Mini_Nbody_HSA_Trace COMMAND ../rocprofv3_mini_nbody_stats_sys_trace.sh )
set_property(TEST RocprofV3_Mini_Nbody_HSA_Trace PROPERTY PASS_REGULAR_EXPRESSION "hsa_system_get_info")
set_property(TEST RocprofV3_Mini_Nbody_HSA_Trace PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME RocprofV3_Mini_Nbody_Memory_Copy_Trace COMMAND ../rocprofv3_mini_nbody_stats_sys_trace.sh )
set_property(TEST RocprofV3_Mini_Nbody_Memory_Copy_Trace PROPERTY PASS_REGULAR_EXPRESSION "MEMORY_COPY_DEVICE_TO_HOST")
set_property(TEST RocprofV3_Mini_Nbody_Memory_Copy_Trace PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# PyTorch tests

add_test(NAME Pytorch_Check_Import COMMAND ../pytorch_check_import.sh )
set_property(TEST Pytorch_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST Pytorch_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Torchvision_Check_Import COMMAND ../torchvision_check_import.sh )
set_property(TEST Pytorch_Torchvision_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST Pytorch_Torchvision_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Torchaudio_Check_Import COMMAND ../torchaudio_check_import.sh )
set_property(TEST Pytorch_Torchaudio_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST Pytorch_Torchaudio_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Check_Gpu_Access COMMAND ../pytorch_check_gpu_access.sh )
set_property(TEST Pytorch_Check_Gpu_Access PROPERTY PASS_REGULAR_EXPRESSION "True")
set_property(TEST Pytorch_Check_Gpu_Access PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Check_Gpu_Name COMMAND ../pytorch_check_gpu_name.sh )
set_property(TEST Pytorch_Check_Gpu_Name PROPERTY PASS_REGULAR_EXPRESSION "AMD")
set_property(TEST Pytorch_Check_Gpu_Name PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Check_ROCm_Dependency COMMAND ../pytorch_check_rocm_dependency.sh )
set_property(TEST Pytorch_Check_ROCm_Dependency PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST Pytorch_Check_ROCm_Dependency PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME Pytorch_Unit_Tests COMMAND ../pytorch_unit_tests.py )
#set_property(TEST Pytorch_Unit_Tests PROPERTY PASS_REGULAR_EXPRESSION "47559 passed")
#set_property(TEST Pytorch_Unit_Tests PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Mnist COMMAND ../pytorch_mnist.sh )
set_property(TEST Pytorch_Mnist PROPERTY PASS_REGULAR_EXPRESSION "Train Epoch: 14")
set_property(TEST Pytorch_Mnist PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Profile_Rocprof-compute_ROCm COMMAND ../../MLExamples/PyTorch_Profiling/rocm-compute-profiler/single_process.sh )
set_property(TEST Pytorch_Profile_Rocprof-compute_ROCm PROPERTY PASS_REGULAR_EXPRESSION "2.1 Speed-of-Light")
set_property(TEST Pytorch_Profile_Rocprof-compute_ROCm PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Profile_Rocprof-sys_ROCm COMMAND ../../MLExamples/PyTorch_Profiling/rocm-systems-profiler/rocm-systems-profiler.sh )
set_property(TEST Pytorch_Profile_Rocprof-sys_ROCm PROPERTY PASS_REGULAR_EXPRESSION "perfetto-trace-")
set_property(TEST Pytorch_Profile_Rocprof-sys_ROCm PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Profile_Rocprofv3 COMMAND ../../MLExamples/PyTorch_Profiling/rocprofv3/kernels.sh )
set_property(TEST Pytorch_Profile_Rocprofv3 PROPERTY PASS_REGULAR_EXPRESSION "kernel_stats.csv")
set_property(TEST Pytorch_Profile_Rocprofv3 PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Pytorch_Profile_TorchProfiler COMMAND ../../MLExamples/PyTorch_Profiling/torch-profiler/single_process.sh )
set_property(TEST Pytorch_Profile_TorchProfiler PROPERTY PASS_REGULAR_EXPRESSION "trace_")
set_property(TEST Pytorch_Profile_TorchProfiler PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Python MPI tests

#add_test(NAME MPI4PY_OpenMPI_Check_Import COMMAND ../mpi4py_check_import.sh --module-name openmpi)
#set_property(TEST MPI4PY_OpenMPI_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
#set_property(TEST MPI4PY_OpenMPI_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME MPI4PY_OpenMPI_Check_Import_MPI COMMAND ../mpi4py_check_import_mpi.sh --module-name openmpi)
#set_property(TEST MPI4PY_OpenMPI_Check_Import_MPI PROPERTY PASS_REGULAR_EXPRESSION "Success")
#set_property(TEST MPI4PY_OpenMPI_Check_Import_MPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME MPI4PY_OpenMPI_Cupy COMMAND ../mpi4py_cupy.sh --module-name openmpi)
#set_property(TEST MPI4PY_OpenMPI_Cupy PROPERTY PASS_REGULAR_EXPRESSION "Success")
#set_property(TEST MPI4PY_OpenMPI_Cupy PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME MPI4PY_Standalone_Check_Import COMMAND ../mpi4py_check_import.sh --module-name mpi4py)
set_property(TEST MPI4PY_Standalone_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST MPI4PY_Standalone_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME MPI4PY_Standalone_Check_Import_MPI COMMAND ../mpi4py_check_import_mpi.sh --module-name mpi4py)
set_property(TEST MPI4PY_Standalone_Check_Import_MPI PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST MPI4PY_Standalone_Check_Import_MPI PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME MPI4PY_Standalone_Cupy COMMAND ../mpi4py_cupy.sh --module-name mpi4py)
set_property(TEST MPI4PY_Standalone_Cupy PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST MPI4PY_Standalone_Cupy PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Cupy tests

add_test(NAME Cupy_Check_Import COMMAND ../cupy_check_import.sh )
set_property(TEST Cupy_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST Cupy_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Cupy_Install_Tests COMMAND ../cupy_install_tests.sh )
set_property(TEST Cupy_Install_Tests PROPERTY PASS_REGULAR_EXPRESSION "11 passed")
set_property(TEST Cupy_Install_Tests PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Rocprof-sys tests

add_test(NAME Rocprof-sys_Source_Avail_Check COMMAND ../rocprof-sys-avail_check.sh --version amd-staging)
set_property(TEST Rocprof-sys_Source_Avail_Check PROPERTY PASS_REGULAR_EXPRESSION "Outputting text configuration file ")
set_property(TEST Rocprof-sys_Source_Avail_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_ROCm_Avail_Check COMMAND ../rocprof-sys-avail_check.sh )
set_property(TEST Rocprof-sys_ROCm_Avail_Check PROPERTY PASS_REGULAR_EXPRESSION "Outputting text configuration file ")
set_property(TEST Rocprof-sys_ROCm_Avail_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_Source_Instrument_Check COMMAND ../rocprof-sys-instrument_check.sh --version amd-staging)
set_property(TEST Rocprof-sys_Source_Instrument_Check PROPERTY PASS_REGULAR_EXPRESSION "The instrumented executable image is stored in ")
set_property(TEST Rocprof-sys_Source_Instrument_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_ROCm_Instrument_Check COMMAND ../rocprof-sys-instrument_check.sh )
set_property(TEST Rocprof-sys_ROCm_Instrument_Check PROPERTY PASS_REGULAR_EXPRESSION "The instrumented executable image is stored in ")
set_property(TEST Rocprof-sys_ROCm_Instrument_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_Source_Run_Check COMMAND ../rocprof-sys-run_check.sh --version amd-staging)
set_property(TEST Rocprof-sys_Source_Run_Check PROPERTY PASS_REGULAR_EXPRESSION "Tracing session 1 ended, total sessions:0")
set_property(TEST Rocprof-sys_Source_Run_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_ROCm_Run_Check COMMAND ../rocprof-sys-run_check.sh )
set_property(TEST Rocprof-sys_ROCm_Run_Check PROPERTY PASS_REGULAR_EXPRESSION "Tracing session 1 ended, total sessions:0")
set_property(TEST Rocprof-sys_ROCm_Run_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_Source_Stream_Overlap COMMAND ../rocprof-sys_stream_overlap.sh --version amd-staging)
set_property(TEST Rocprof-sys_Source_Stream_Overlap PROPERTY PASS_REGULAR_EXPRESSION "proto")
set_property(TEST Rocprof-sys_Source_Stream_Overlap PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-sys_ROCm_Stream_Overlap COMMAND ../rocprof-sys_stream_overlap.sh )
set_property(TEST Rocprof-sys_ROCm_Stream_Overlap PROPERTY PASS_REGULAR_EXPRESSION "proto")
set_property(TEST Rocprof-sys_ROCm_Stream_Overlap PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Rocprof-compute Tests

add_test(NAME Rocprof-compute_Source_Roofline_Check COMMAND ../rocprof-compute_roofline_check.sh --version develop)
set_property(TEST Rocprof-compute_Source_Roofline_Check PROPERTY PASS_REGULAR_EXPRESSION "Empirical Roofline PDFs saved!")
set_property(TEST Rocprof-compute_Source_Roofline_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Rocprof-compute_ROCm_Roofline_Check COMMAND ../rocprof-compute_roofline_check.sh)
set_property(TEST Rocprof-compute_ROCm_Roofline_Check PROPERTY PASS_REGULAR_EXPRESSION "Empirical Roofline PDFs saved!")
set_property(TEST Rocprof-compute_ROCm_Roofline_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "disabled on MI300")

add_test(NAME Rocprof-compute_Source_Profile_Check COMMAND ../rocprof-compute_profile_check.sh --version develop)
set_property(TEST Rocprof-compute_Source_Profile_Check PROPERTY PASS_REGULAR_EXPRESSION "Skipping roofline" "Roofline temporarily disabled in MI300")
set_property(TEST Rocprof-compute_Source_Profile_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
set_property(TEST Rocprof-compute_Source_Profile_Check PROPERTY TIMEOUT 400)

add_test(NAME Rocprof-compute_ROCm_Profile_Check COMMAND ../rocprof-compute_profile_check.sh)
set_property(TEST Rocprof-compute_ROCm_Profile_Check PROPERTY PASS_REGULAR_EXPRESSION "Skipping roofline" "Roofline temporarily disabled in MI300")
set_property(TEST Rocprof-compute_ROCm_Profile_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")


add_test(NAME Rocprof-compute_Source_Analyze_Check COMMAND ../rocprof-compute_analyze_check.sh --version develop)
set_property(TEST Rocprof-compute_Source_Analyze_Check PROPERTY PASS_REGULAR_EXPRESSION "7.1 Wavefront Launch Stats")
set_property(TEST Rocprof-compute_Source_Analyze_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
set_property(TEST Rocprof-compute_Source_Analyze_Check PROPERTY TIMEOUT 400)

add_test(NAME Rocprof-compute_ROCm_Analyze_Check COMMAND ../rocprof-compute_analyze_check.sh)
set_property(TEST Rocprof-compute_ROCm_Analyze_Check PROPERTY PASS_REGULAR_EXPRESSION "7.1 Wavefront Launch Stats")
set_property(TEST Rocprof-compute_ROCm_Analyze_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# HPCToolkit Tests

add_test(NAME HPCToolkit_Check_ROCm_Dependency COMMAND ../hpctoolkit_check_rocm_dependency.sh)
set_property(TEST HPCToolkit_Check_ROCm_Dependency PROPERTY PASS_REGULAR_EXPRESSION "AMD ROCm support: yes")
set_property(TEST HPCToolkit_Check_ROCm_Dependency PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HPCToolkit_HPCRun_Check COMMAND ../hpctoolkit_check_hpcrun.sh)
set_property(TEST HPCToolkit_HPCRun_Check PROPERTY PASS_REGULAR_EXPRESSION "hpcrun")
set_property(TEST HPCToolkit_HPCRun_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HPCToolkit_HPCStruct_Check COMMAND ../hpctoolkit_check_hpcstruct.sh)
set_property(TEST HPCToolkit_HPCStruct_Check PROPERTY PASS_REGULAR_EXPRESSION "gpubins-used")
set_property(TEST HPCToolkit_HPCStruct_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HPCToolkit_HPCProf_Check COMMAND ../hpctoolkit_check_hpcprof.sh)
set_property(TEST HPCToolkit_HPCProf_Check PROPERTY PASS_REGULAR_EXPRESSION "metrics")
set_property(TEST HPCToolkit_HPCProf_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HPCToolkit_HPCViewer_Check COMMAND ../hpctoolkit_check_hpcviewer.sh)
set_property(TEST HPCToolkit_HPCViewer_Check PROPERTY PASS_REGULAR_EXPRESSION "Java version ")
set_property(TEST HPCToolkit_HPCViewer_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# JAX Tests

add_test(NAME JAX_Check_Import COMMAND ../jax_check_import.sh )
set_property(TEST JAX_Check_Import PROPERTY PASS_REGULAR_EXPRESSION "Success")
set_property(TEST JAX_Check_Import PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME JAX_Check_Gpu_Access COMMAND ../jax_check_gpu_access.sh )
set_property(TEST JAX_Check_Gpu_Access PROPERTY PASS_REGULAR_EXPRESSION "RocmDevice")
set_property(TEST JAX_Check_Gpu_Access PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME JAX_Mnist COMMAND ../jax_mnist.sh )
set_property(TEST JAX_Mnist PROPERTY PASS_REGULAR_EXPRESSION "Epoch 9 in")
set_property(TEST JAX_Mnist PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#add_test(NAME JAX_Check_Performance COMMAND ../jax_check_performance.sh )
#set_property(TEST JAX_Check_Performance PROPERTY PASS_REGULAR_EXPRESSION "Acheived bfloat16 TFLOPS: 129")
#set_property(TEST JAX_Check_Performance PROPERTY SKIP_REGULAR_EXPRESSION "module spider")


# TAU Tests

add_test(NAME TAU_Trace_Check COMMAND ../tau_exec_check.sh --tau-trace)
set_property(TEST TAU_Trace_Check PROPERTY PASS_REGULAR_EXPRESSION "tautrace.0")
set_property(TEST TAU_Trace_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider tau")

add_test(NAME TAU_Profile_Check COMMAND ../tau_exec_check.sh --tau-profile)
set_property(TEST TAU_Profile_Check PROPERTY PASS_REGULAR_EXPRESSION "profile.0")
set_property(TEST TAU_Profile_Check PROPERTY SKIP_REGULAR_EXPRESSION "module spider tau")

add_test(NAME TAU_Check_HIP_Profile COMMAND ../tau_exec_check.sh --tau-profile)
set_property(TEST TAU_Check_HIP_Profile PROPERTY PASS_REGULAR_EXPRESSION "hipMemcpy")
set_property(TEST TAU_Check_HIP_Profile PROPERTY SKIP_REGULAR_EXPRESSION "module spider tau")

add_test(NAME TAU_Check_MPI_Profile COMMAND ../tau_exec_check.sh --tau-profile)
set_property(TEST TAU_Check_MPI_Profile PROPERTY PASS_REGULAR_EXPRESSION "MPI_Allreduce()")
set_property(TEST TAU_Check_MPI_Profile PROPERTY SKIP_REGULAR_EXPRESSION "module spider tau")


# Score-P Tests

add_test(NAME Score-P_Check_Version COMMAND ../scorep_check_version.sh)
set_property(TEST Score-P_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "Score-P 9.0-beta")
set_property(TEST Score-P_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Miniconda3 Tests

add_test(NAME Miniconda3_Check_Version COMMAND ../miniconda3_check_version.sh)
set_property(TEST Miniconda3_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "conda 24.9.2")
set_property(TEST Miniconda3_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Miniconda3_Env_Create COMMAND ../miniconda3_env_create.sh)
set_property(TEST Miniconda3_Env_Create PROPERTY PASS_REGULAR_EXPRESSION "env_for_test          *")
set_property(TEST Miniconda3_Env_Create PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Miniconda3_Env_Install COMMAND ../miniconda3_env_install.sh)
set_property(TEST Miniconda3_Env_Install PROPERTY PASS_REGULAR_EXPRESSION "jaxlib")
set_property(TEST Miniconda3_Env_Install PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

# Miniforge3 Tests

add_test(NAME Miniforge3_Check_Conda_Version COMMAND ../miniforge3_check_version.sh)
set_property(TEST Miniforge3_Check_Conda_Version PROPERTY PASS_REGULAR_EXPRESSION "conda 24.9.0")
set_property(TEST Miniforge3_Check_Conda_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Miniforge3_Check_Mamba_Version COMMAND ../miniforge3_check_version.sh)
set_property(TEST Miniforge3_Check_Mamba_Version PROPERTY PASS_REGULAR_EXPRESSION "mamba 1.5.9")
set_property(TEST Miniforge3_Check_Mamba_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Miniforge3_Env_Create COMMAND ../miniforge3_env_create.sh)
set_property(TEST Miniforge3_Env_Create PROPERTY PASS_REGULAR_EXPRESSION "env_for_test          *")
set_property(TEST Miniforge3_Env_Create PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Miniforge3_Env_Install COMMAND ../miniforge3_env_install.sh)
set_property(TEST Miniforge3_Env_Install PROPERTY PASS_REGULAR_EXPRESSION "jaxlib")
set_property(TEST Miniforge3_Env_Install PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

#HDF5 Tests

add_test(NAME HDF5_Check_Version COMMAND ../hdf5_check_version.sh)
set_property(TEST HDF5_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "h5dump: Version 1.14.5")
set_property(TEST HDF5_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME HDF5_Check_H5Dump COMMAND ../hdf5_check_h5dump.sh)
set_property(TEST HDF5_Check_H5Dump PROPERTY PASS_REGULAR_EXPRESSION "391, 392, 393, 394, 395, 396, 397, 398, 399, 400")
set_property(TEST HDF5_Check_H5Dump PROPERTY SKIP_REGULAR_EXPRESSION "module spider hdf5" "Skip")

#NETCDF Tests

add_test(NAME Netcdf_C_Check_Version COMMAND ../netcdf-c_check_version.sh)
set_property(TEST Netcdf_C_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "netcdf library version 4.9.3-rc1")
set_property(TEST Netcdf_C_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_C_Check_Parallel_On COMMAND ../netcdf-c_print_config.sh)
set_property(TEST Netcdf_C_Check_Parallel_On PROPERTY PASS_REGULAR_EXPRESSION "--has-parallel     -> yes")
set_property(TEST Netcdf_C_Check_Parallel_On PROPERTY SKIP_REGULAR_EXPRESSION "module spider" "Skip")

add_test(NAME Netcdf_C_Check_HDF5_On COMMAND ../netcdf-c_print_config.sh)
set_property(TEST Netcdf_C_Check_HDF5_On PROPERTY PASS_REGULAR_EXPRESSION "--has-hdf5         -> yes")
set_property(TEST Netcdf_C_Check_HDF5_On PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_C_Check_Pres_Temp_4D_WR COMMAND ../netcdf-c_pres_temp_4D.sh)
set_property(TEST Netcdf_C_Check_Pres_Temp_4D_WR PROPERTY PASS_REGULAR_EXPRESSION "SUCCESS writing example file pres_temp_4D.nc!")
set_property(TEST Netcdf_C_Check_Pres_Temp_4D_WR PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_C_Check_Pres_Temp_4D_RD COMMAND ../netcdf-c_pres_temp_4D.sh)
set_property(TEST Netcdf_C_Check_Pres_Temp_4D_RD PROPERTY PASS_REGULAR_EXPRESSION "SUCCESS reading example file pres_temp_4D.nc!")
set_property(TEST Netcdf_C_Check_Pres_Temp_4D_RD PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_C_Check_Parallel_IO COMMAND ../netcdf-c_parallel_vara.sh)
set_property(TEST Netcdf_C_Check_Parallel_IO PROPERTY PASS_REGULAR_EXPRESSION "0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,")
set_property(TEST Netcdf_C_Check_Parallel_IO PROPERTY SKIP_REGULAR_EXPRESSION "module spider netcdf-c" "Skip")

add_test(NAME Netcdf_Fortran_Check_Version COMMAND ../netcdf-fortran_check_version.sh)
set_property(TEST Netcdf_Fortran_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "4.6.2-rc1")
set_property(TEST Netcdf_Fortran_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_Fortran_Check_Pres_Temp_4D_WR COMMAND ../netcdf-fortran_pres_temp_4D.sh)
set_property(TEST Netcdf_Fortran_Check_Pres_Temp_4D_WR PROPERTY PASS_REGULAR_EXPRESSION "SUCCESS writing example file pres_temp_4D.nc!")
set_property(TEST Netcdf_Fortran_Check_Pres_Temp_4D_WR PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_Fortran_Check_Pres_Temp_4D_RD COMMAND ../netcdf-fortran_pres_temp_4D.sh)
set_property(TEST Netcdf_Fortran_Check_Pres_Temp_4D_RD PROPERTY PASS_REGULAR_EXPRESSION "SUCCESS reading example file pres_temp_4D.nc!")
set_property(TEST Netcdf_Fortran_Check_Pres_Temp_4D_RD PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME Netcdf_Fortran_Check_Parallel_IO COMMAND ../netcdf-fortran_simple_xy_par.sh)
set_property(TEST Netcdf_Fortran_Check_Parallel_IO PROPERTY PASS_REGULAR_EXPRESSION "3, 3, 3, 3 ;")
set_property(TEST Netcdf_Fortran_Check_Parallel_IO PROPERTY SKIP_REGULAR_EXPRESSION "module spider netcdf-fortran" "module spider netcdf-c" "Skip")

# FFTW Tests

add_test(NAME FFTW_Check_Version COMMAND ../fftw_check_version.sh)
set_property(TEST FFTW_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "fftw-wisdom tool for FFTW version 3.3.10.")
set_property(TEST FFTW_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME FFTWF_Check_Version COMMAND ../fftwf_check_version.sh)
set_property(TEST FFTWF_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "fftw-wisdom tool for FFTW version 3.3.10.")
set_property(TEST FFTWF_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")

add_test(NAME FFTWL_Check_Version COMMAND ../fftwl_check_version.sh)
set_property(TEST FFTWL_Check_Version PROPERTY PASS_REGULAR_EXPRESSION "fftw-wisdom tool for FFTW version 3.3.10.")
set_property(TEST FFTWL_Check_Version PROPERTY SKIP_REGULAR_EXPRESSION "module spider")
