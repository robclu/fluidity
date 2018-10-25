#==--- CMakeLists.txt -------------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fluidity VERSION 0.1.0 LANGUAGES CXX)

#==--- Build Type -----------------------------------------------------------==#

if(NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
endif(NOT CMAKE_BUILD_TYPE)

#==--- Compiler paths -------------------------------------------------------==#

if (APPLE)
else()
  set(CUDA_COMPILER_PATH  )
  set(GCC_COMPILER_PATH   )
  set(CLANG_COMPILER_PATH "/lsc/opt/clang-6.0/bin/clang++")
endif()

# Set clang usage variable for cuda:
if (NOT ${CUDA_CLANG})
  set(CUDA_CLANG FALSE)
else()
  set(CUDA_CLANG TRUE)
endif()

# Set clang usage variable for cxx:
if (NOT ${CXX_CLANG})
  set(CXX_CLANG FALSE)
else()
  set(CXX_CLANG TRUE)
endif()

message("-- Clang cuda compiler option : ${CUDA_CLANG}")
message("-- Clang cxx  compiler option : ${CXX_CLANG}")

# If clang for CXX then set the CXX compiler:
if (${CXX_CLANG})
  if (NOT EXISTS ${CLANG_COMPILER_PATH})
    message("-- CXX : Using system clang++ for cxx since CXX_CLANG=ON"   )
    message("         but CLANG_PATH is empty. Specify CLANG_PATH=<path>")
    message("         in CMAKE build, or set it in CMakeLists.txt"       )
    set(FLUID_CXX_COMPILER clang++)
  else() 
    set(FLUID_CXX_COMPILER ${CLANG_COMPILER_PATH})
  endif()
endif()

# If clang for cuda then set the cuda compiler. When clang is used for cuda,
# the CXX compiler is automatically set to clang as well.
if (${CUDA_CLANG})
  if (NOT EXISTS ${CLANG_COMPILER_PATH})
    message("-- CUDA : Using system clang++ for cuda since -DCUDA_CLANG=ON" )
    message("          but CLANG_PATH is empty. Specify -DCLANG_PATH=<path>")
    message("          in CMAKE build, or set it in CMakeLists.txt"         )
    set(CLANG_COMPILER_PATH clang++)
  endif() 
  set(FLUID_CUDA_COMPILER ${CLANG_COMPILER_PATH})

  # If we use clang for cuda, we use clang for c++ as well
  set(FLUID_CXX_COMPILER ${CLANG_COMPILER_PATH})
  set(CUDA_PATH "/lsc/opt/cuda-9.0")
  set(USING_CLANG TRUE)
else()
  set(USING_CLANG FALSE)
endif()

if (NOT EXISTS ${CUDA_PATH})
  set(CUDA_PATH "/lsc/opt/cuda-9.2")
  message("-- CUDA : CUDA_PATH is not valid, using : ${CUDA_PATH}")
endif()
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})

if (NOT ${USING_CLANG})
  #set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
  set(FLUID_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(FLUID_CXX_COMPILER   ${FLUID_CUDA_COMPILER})
endif()

#string(REGEX MATCH "clang" CLANG_REGEX ${CMAKE_CXX_COMPILER})
#string(COMPARE EQUAL "${CLANG_REGEX}" "clang" USING_CLANG)

# Test that the version of the cuda compiler is sufficient:
message("-- Using CUDA compiler : ${FLUID_CUDA_COMPILER}")
message("-- Using CXX  compiler : ${FLUID_CXX_COMPILER}")
execute_process(
  COMMAND ${FLUID_CUDA_COMPILER} --version OUTPUT_VARIABLE NVCC_VERSION
)

if (NVCC_VERSION MATCHES "V9")
else()
  if (NOT ${USING_CLANG})
    message("-- ERROR: NVCC Version > 9 is required.")
    message(
      FATAL_ERROR "     Please set -DCUDA_PATH=<path/to/cuda/root> > V9"
    )
  endif()
endif()

#==--- Include directories --------------------------------------------------==#

# Gtest does not currently build without this flag!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

#if (EXISTS ${fluidity_SOURCE_DIR}/googletest)
#  add_subdirectory(googletest)
#else()
#  message("-- NOTE: Googletest not found, fetching ...")
#  execute_process(
#    COMMAND git clone https://github.com/google/googletest.git 
#      ${fluidity_SOURCE_DIR}/googletest
#  )
#  message("-- DONE: Fetched googletest.")
#  add_subdirectory(googletest)
#endif()

#==--- Compiler Flags -------------------------------------------------------==#

set(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)
if (APPLE)
  set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
else()
  set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

message("-- CUDA INCLUDE DIR: ${CUDA_INCLUDE_DIR}")
message("-- CUDA LIBRARY DIR: ${CUDA_LIB_DIRS}"   )

set(CUDA_DEFS "-DFLUIDITY_CUDA_AVAILABLE=1")
set(CUDA_ARCH_VER "35")

if (NOT MAX_REG_COUNT)
  set(MAX_REG_COUNT 64)
endif()

if(WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall")
  set(FLAGS_PERF    "-O3 -std=c++14 -pthread -Wno-c++1z-extensions")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(DEBUG_FLAGS   "${DEBUG_FLAGS} -Wno-unused-parameter -lineinfo")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  if (NOT ${USING_CLANG})
    set(CUDA_ARCH_FLAG "-gencode arch=compute_${CUDA_ARCH_VER},")
    set(CUDA_ARCH_FLAG "${CUDA_ARCH_FLAG}code=sm_${CUDA_ARCH_VER}")
    set(CUDA_EXP_FLAGS "-expt-relaxed-constexpr -expt-extended-lambda")
    set(CUDA_FLAGS     "-use_fast_math -O3 -std=c++14 ${CUDA_EXP_FLAGS}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -maxrregcount=${MAX_REG_COUNT}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} ${CUDA_ARCH_FLAG}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -lineinfo -Xptxas=-v")
  else()
    set(CUDA_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}") 
    set(CUDA_FLAGS "${CUDA_FLAGS} --cuda-gpu-arch=sm_${CUDA_ARCH_VER}")
    set(CUDA_FLAGS "${CUDA_FLAGS} -O3 -std=c++14")
  endif()
endif()
set(CUDA_FLAGS "${CUDA_FLAGS} ${CUDA_DEFS}")

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

add_subdirectory(googletest)

#==--- Subdirectories -------------------------------------------------------==#

add_subdirectory(tests)
add_subdirectory(validation)