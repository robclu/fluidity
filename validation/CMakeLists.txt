#==--- validation/CMakeLists.txt --------------------------------------------==#
#
#                      Copyright (c) 2018 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${fluidity_SOURCE_DIR}/cmake)
include(fluidity_cmake)

set(LIBS -lstdc++-fs)

if (APPLE)
  set(CUDA_LIBRARIES -lcudart -ldl -lrt -lcudadevrt -pthread)
else()
  set(CUDA_LIBRARIES -lcudart_static -ldl -lrt -lcudadevrt -pthread)
endif()

set(SOURCE_DIR ${fluidity_SOURCE_DIR}/validation)

#==--- Directories ----------------------------------------------------------==#
fluid_include_directories(${CUDA_INCLUDE_DIR} ${fluidity_SOURCE_DIR}/include)
fluid_library_directories(${CUDA_LIB_DIRS})

#==--- Valiation tests ------------------------------------------------------==#

# TORO test case 1:
fluid_add_executable(toro_1d_case_1 ${SOURCE_DIR}/toro_1d_case_1_validation.cu)
fluid_target_link_libraries(toro_1d_case_1 ${CUDA_LIBRARIES})

# TORO test case 2:
fluid_add_executable(toro_1d_case_2 ${SOURCE_DIR}/toro_1d_case_2_validation.cu)
fluid_target_link_libraries(toro_1d_case_2 ${CUDA_LIBRARIES})

# TORO test case 3:
fluid_add_executable(toro_1d_case_3 ${SOURCE_DIR}/toro_1d_case_3_validation.cu)
fluid_target_link_libraries(toro_1d_case_3 ${CUDA_LIBRARIES})

# TORO test case 4:
fluid_add_executable(toro_1d_case_4 ${SOURCE_DIR}/toro_1d_case_4_validation.cu)
fluid_target_link_libraries(toro_1d_case_4 ${CUDA_LIBRARIES})

# TORO test case 5:
fluid_add_executable(toro_1d_case_5 ${SOURCE_DIR}/toro_1d_case_5_validation.cu)
fluid_target_link_libraries(toro_1d_case_5 ${CUDA_LIBRARIES})

#==--- Finish Up ------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
fluid_create_all_targets()