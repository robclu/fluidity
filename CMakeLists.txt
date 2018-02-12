#==--- CMakeLists.txt -------------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fluidity VERSION 0.1.0 LANGUAGES CXX CUDA)

#==--- Build Type -----------------------------------------------------------==#

if(NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
endif(NOT CMAKE_BUILD_TYPE)

# Change if we want to use clang as the host compiler:
if (NOT APPLE)
  set(CMAKE_CUDA_COMPILER /lsc/opt/cuda-9.0/bin/nvcc)
endif()

# Test that the version of the cuda compiler is sufficient:
message("-- Using CUDA Compiler : ${CMAKE_CUDA_COMPILER}")
execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE NVCC_VERSION
)

if (NVCC_VERSION MATCHES "V9")
else()
  message("-- ERROR: NVCC Version > 9 is required.")
  message(
    FATAL_ERROR "     Please set -DCMAKE_CUDA_COMPILER=<path/to/nvcc/version>9>"
  )
endif()

#==--- Include directories --------------------------------------------------==#

message("CUDA: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(
  ${fluidity_SOURCE_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  /home/raid/rjc201/opt/google/include
)

#==--- Compiler Flags -------------------------------------------------------==#

if(WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall")
  set(FLAGS_PERF    "-O3 -std=c++14 -pthread")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CUDA_FLAGS    "-use_fast_math -O3 -std=c++14 -arch=sm_30 -pthread")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}" 
    CACHE STRING "CXX FLAGS" FORCE
  )
else()
  set(
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
    CACHE STRING "CXX FLAGS" FORCE
  )
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")

#==--- Subdirectories -------------------------------------------------------==#

add_subdirectory(src)
add_subdirectory(tests)