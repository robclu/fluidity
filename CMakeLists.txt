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
set(CMAKE_CUDA_COMPILER /lsc/opt/cuda-9.1/bin/nvcc)

#==--- Include directories --------------------------------------------------==#

include_directories(${fluidity_SOURCE_DIR}/include)

#==--- Compiler Flags -------------------------------------------------------==#

if(WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall")
  set(FLAGS_PERF    "-O3 -std=c++14")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CUDA_FLAGS    "-use_fast_math -O3 -std=c++14")
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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}"
    CACHE STRING "CUDA FLAGS" FORCE)

#==--- Subdirectories -------------------------------------------------------==#

add_subdirectory(src)
add_subdirectory(tests)