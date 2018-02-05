#==--- src/CMakeLists.txt ---------------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

add_library(
  fluditiy_cuda_test STATIC
  fluidity_cuda_test.cpp
  fluidity_cuda_test.cu
)

# Set the CUDA architecture.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")

# Make sure that the test is built with C++14 support. Marking this as PUBLIC
# means that anything which links to the library will need C++14 support too.
target_compile_features(fluditiy_cuda_test PUBLIC cxx_std_14)

# Need to explicity state that all the CUDA files in the library to be built
# with -dc so that the member functions can be called by other libraries and
# executables.
set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)