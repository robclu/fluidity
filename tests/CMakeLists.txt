#==--- src/CMakeLists.txt ---------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(LINK_DIRECTORIES ${LINK_DIRECTORIES} /home/raid/rjc201/opt/google/lib)

add_executable(cuda_test cuda_test.cu)
set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test PRIVATE fluidity_cuda_test)


#==--- Tests ----------------------------------------------------------------==#

add_executable(container_hosts_tests container_tests_host.cpp)
target_link_libraries(container_hosts_tests -lgtest -lgtest_main)

#if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  #set_property(
  #  TARGET test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  #)
#endif()