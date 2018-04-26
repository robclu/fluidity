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
  #set(CMAKE_CUDA_COMPILER /lsc/opt/clang-6.0/bin/clang++)
endif()

string(REGEX MATCH "clang" CLANG_REGEX ${CMAKE_CUDA_COMPILER})
string(COMPARE EQUAL "${CLANG_REGEX}" "clang" USING_CLANG)

# Test that the version of the cuda compiler is sufficient:
message("-- Using CUDA Compiler : ${CMAKE_CUDA_COMPILER}")
execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE NVCC_VERSION
)

if (NVCC_VERSION MATCHES "V9")
else()
  if (NOT ${USING_CLANG})
    message("-- ERROR: NVCC Version > 9 is required.")
    message(
      FATAL_ERROR "     Please set -DCMAKE_CUDA_COMPILER=<path/to/nvcc/version>9>"
    )
  endif()
endif()

#==--- Include directories --------------------------------------------------==#

message("-- CUDA INCLUDE DIR: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Gtest does not currently build without this flag!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

if (EXISTS ${fluidity_SOURCE_DIR}/googletest)
  add_subdirectory(googletest)
else()
  message("-- NOTE: Googletest not found, fetching ...")
  execute_process(
    COMMAND git clone https://github.com/google/googletest.git 
      ${fluidity_SOURCE_DIR}/googletest
  )
  message("-- DONE: Fetched googletest.")
  add_subdirectory(googletest)
endif()

include_directories(
  ${fluidity_SOURCE_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#==--- Compiler Flags -------------------------------------------------------==#

if(WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall")
  set(FLAGS_PERF    "-O3 -std=c++14 -pthread -Wno-c++1z-extensions")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(DEBUG_FLAGS   "${DEBUG_FLAGS} -Wno-unused-parameter")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  if (NOT ${USING_CLANG})
    set(CUDA_EXP_FLAGS "-expt-relaxed-constexpr -expt-extended-lambda")
    set(CUDA_FLAGS    "-use_fast_math -O3 -std=c++14 ${CUDA_EXP_FLAGS}")
  else()
    set(CUDA_FLAGS "--cuda-path=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") 
    set(CUDA_FLAGS "${CUDA_FLAGS} --cuda-gpu-arch=sm_50") 
  endif()
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

if (NOT ${USING_CLANG})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_FLAGS}")
endif()

# NVIDIA IS SO USELESS THAT IT CANNOT SUPPORT ANY NON-ARCHAIC COMPILER VERSION
# SO WE CANNOT USE ANY NEW c++ FEATURES, PROBABLY FOR THE NEXT 10 YEARS!! =/
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${CMAKE_CXX_FLAGS}")

#==--- Subdirectories -------------------------------------------------------==#

#add_subdirectory(src)
add_subdirectory(tests)