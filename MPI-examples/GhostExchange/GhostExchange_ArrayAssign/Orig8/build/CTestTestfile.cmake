# CMake generated Testfile for 
# Source directory: /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2
# Build directory: /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SimpleTest "/share/contrib-modules/openmpi/ompi5.0.3-ucc1.3.x-ucx1.16.x-rocm6.2.0/bin/mpiexec" "-n" "128" "/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/GhostExchange")
set_tests_properties(SimpleTest PROPERTIES  _BACKTRACE_TRIPLES "/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/CMakeLists.txt;39;add_test;/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/CMakeLists.txt;0;")
