#==--- tests/CMakeLists.txt -------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${fluidity_SOURCE_DIR}/cmake)
include(fluidity_cmake)

if (EXISTS ${fluidity_SOURCE_DIR}/googletest)
  set(GTEST_INCLUDE  ${fluidity_SOURCE_DIR}/googletest/googletest/include)
  set(GTEST_LIB_DIRS ${fluidity_SOURCE_DIR}/build/googletest/googlemock/gtest)
  if (CMAKE_BUILD_TYPE EQUAL "Debug")
    set(GTEST_BOTH_LIBRARIES -lgtestd -lgtest_maind)
  else()
    set(GTEST_BOTH_LIBRARIES -lgtest -lgtest_main)
  endif()
else()
  set(GTEST_INCLUDE )
endif()

set(LIBS -lstdc++-fs)

#add_executable(cuda_test cuda_test.cu)
#set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test PRIVATE fluidity_cuda_test)

if (APPLE)
  set(CUDA_LIBRARIES -lcudart -ldl -lrt -lcudadevrt -pthread)
else()
  set(CUDA_LIBRARIES -lcudart_static -ldl -lrt -lcudadevrt -pthread)
endif()

#fluid_add_definitions(${CMAKE_CUDA_FLAGS})
fluid_include_directories(
  ${CUDA_INCLUDE_DIR}
  ${fluidity_SOURCE_DIR}/include
  ${GTEST_INCLUDE}
)
fluid_library_directories(${CUDA_LIB_DIRS} ${GTEST_LIB_DIRS})

#==--- Tests ----------------------------------------------------------------==#

fluid_add_executable(
  algorithm_host_tests
  ${fluidity_SOURCE_DIR}/tests/algorithm_tests_host.cpp
)
fluid_target_link_libraries(algorithm_host_tests ${GTEST_BOTH_LIBRARIES})

fluid_add_executable(
  algorithm_device_tests
  ${fluidity_SOURCE_DIR}/tests/algorithm_tests_device.cu
)
fluid_target_link_libraries(
  algorithm_device_tests
  ${GTEST_BOTH_LIBRARIES}
  ${CUDA_LIBRARIES}
)

#add_executable(container_device_tests container_tests_device.cu)
#set_target_properties(
#  container_device_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#)
#target_link_libraries(container_device_tests ${GTEST_BOTH_LIBRARIES})

#add_executable(container_host_tests container_tests_host.cu)
#set_target_properties(
#  container_host_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#)
#target_link_libraries(container_host_tests ${GTEST_BOTH_LIBRARIES})
#set_property(
# TARGET container_host_tests
#  PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
#)

fluid_add_executable(
  dimension_host_tests
  ${fluidity_SOURCE_DIR}/tests/dimension_tests_host.cpp
)
fluid_target_link_libraries(dimension_host_tests ${GTEST_BOTH_LIBRARIES})

fluid_add_executable(
  iterator_host_tests
  ${fluidity_SOURCE_DIR}/tests/iterator_tests_host.cpp
)
fluid_target_link_libraries(iterator_host_tests ${GTEST_BOTH_LIBRARIES})

fluid_add_executable(
  container_device_tests
  ${fluidity_SOURCE_DIR}/tests/container_tests_device.cu
)
fluid_target_link_libraries(
  container_device_tests
  ${GTEST_BOTH_LIBRARIES}
  ${CUDA_LIBRARIES}
)

fluid_add_executable(
  levelset_tests
  ${fluidity_SOURCE_DIR}/tests/levelset_tests.cu
)
fluid_target_link_libraries(
  levelset_tests
  ${GTEST_BOTH_LIBRARIES}
  ${CUDA_LIBRARIES}
)

fluid_add_executable(
  reconstruction_tests
  ${fluidity_SOURCE_DIR}/tests/reconstruction_tests.cpp
)
fluid_target_link_libraries(
  reconstruction_tests
  ${GTEST_BOTH_LIBRARIES}
  ${CUDA_LIBRARIES}
)

fluid_add_executable(
  setting_tests
  ${fluidity_SOURCE_DIR}/tests/setting_tests.cpp
)
fluid_target_link_libraries(setting_tests ${GTEST_BOTH_LIBRARIES})

fluid_add_executable(
  simulator_tests
  ${fluidity_SOURCE_DIR}/tests/simulator_tests.cu
)
fluid_target_link_libraries(
  simulator_tests
  ${CUDA_LIBRARIES}
)

fluid_add_executable(
  state_tests_host
  ${fluidity_SOURCE_DIR}/tests/state_tests_host.cpp
)
fluid_target_link_libraries(state_tests_host ${GTEST_BOTH_LIBRARIES})

#add_executable(solver_host_tests solver_tests_host.cpp)
#target_link_libraries(solver_host_tests ${GTEST_BOTH_LIBRARIES})

#add_executable(state_host_tests state_tests_host.cpp)
#target_link_libraries(state_host_tests ${GTEST_BOTH_LIBRARIES})

#if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  #set_property(
  #  TARGET test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  #)
#endif()

#==--- Finish Up ------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
fluid_create_all_targets()