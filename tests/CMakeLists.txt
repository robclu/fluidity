#==--- src/CMakeLists.txt ---------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

#add_executable(cuda_test cuda_test.cu)
#set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test PRIVATE fluidity_cuda_test)

# Define the google test libraries to link against:
set(GTEST_BOTH_LIBRARIES gtest gtest_main)

#==--- Tests ----------------------------------------------------------------==#

add_executable(algorithm_host_tests algorithm_tests_host.cpp)
set_target_properties(
  algorithm_host_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(algorithm_host_tests ${GTEST_BOTH_LIBRARIES})

add_executable(algorithm_device_tests algorithm_tests_device.cu)
set_target_properties(
  algorithm_device_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(algorithm_device_tests ${GTEST_BOTH_LIBRARIES})

add_executable(container_device_tests container_tests_device.cu)
set_target_properties(
  container_device_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(container_device_tests ${GTEST_BOTH_LIBRARIES})

add_executable(container_host_tests container_tests_host.cpp)
target_link_libraries(container_host_tests ${GTEST_BOTH_LIBRARIES})

add_executable(dimension_host_tests dimension_tests_host.cpp)
target_link_libraries(dimension_host_tests ${GTEST_BOTH_LIBRARIES})

add_executable(iterator_host_tests iterator_tests_host.cpp)
target_link_libraries(iterator_host_tests ${GTEST_BOTH_LIBRARIES})

add_executable(simulator_tests simulator_tests.cpp)
target_link_libraries(simulator_tests ${GTEST_BOTH_LIBRARIES})

add_executable(state_host_tests state_tests_host.cpp)
target_link_libraries(state_host_tests ${GTEST_BOTH_LIBRARIES})

#if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  #set_property(
  #  TARGET test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  #)
#endif()