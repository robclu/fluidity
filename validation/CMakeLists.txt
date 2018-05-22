#==--- validation/CMakeLists.txt --------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

# Include the cmake scripts which allow clang, nvcc, and gcc to all be used to
# compile the validation cases.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${fluidity_SOURCE_DIR}/cmake)
include(fluidity_cmake)

if (EXISTS ${fluidity_SOURCE_DIR}/googletest)
  set(GTEST_INCLUDE ${fluidity_SOURCE_DIR}/googletest/googletest/include)
else()
  set(GTEST_INCLUDE )
endif()

if (APPLE)
  set(CUDA_LIBRARIES -lcudart -ldl -lrt -lcudadevrt -pthread)
else()
  set(CUDA_LIBRARIES -lcudart_static -ldl -lrt -lcudadevrt -pthread)
endif()

if (NOT ${USING_CLANG})
else()
  set(LIBRARIES -lstdc++fs)
endif()

# Set the include and library paths for the validation cases.
fluid_include_directories(
  ${CUDA_INCLUDE_DIR}
  ${fluidity_SOURCE_DIR}/include
  ${GTEST_INCLUDE}
)
fluid_library_directories(${CUDA_LIB_DIRS})

#==--- Tests ----------------------------------------------------------------==#

fluid_add_executable(
  toro_1d_case_1_validation
  ${fluidity_SOURCE_DIR}/validation/toro_1d_case_1_validation.cu
)
fluid_target_link_libraries(
  toro_1d_case_1_validation
  ${LIBRARIES}
  ${CUDA_LIBRARIES}
  gtest_main
)

#if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  #set_property(
  #  TARGET test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  #)
#endif()