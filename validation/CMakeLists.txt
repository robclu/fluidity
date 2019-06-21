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
  set(CUDA_LIBRARIES -lcudart_static -ldl -lrt -lcudadevrt)
endif()

if (${USING_CLANG})
  set(CUDA_LIBRARIES "${CUDA_LIBRARIES} -pthread")
endif()

set(SOURCE_DIR ${fluidity_SOURCE_DIR}/validation)

#==--- Directories ----------------------------------------------------------==#
fluid_include_directories(${CUDA_INCLUDE_DIR} ${fluidity_SOURCE_DIR}/include)
fluid_library_directories(${CUDA_LIB_DIRS})

#==--- Valiation tests ------------------------------------------------------==#

#==--- [1D tests] -----------------------------------------------------------==#

# TORO test case 1:
fluid_add_executable(1d_toro_case_1 ${SOURCE_DIR}/1d_toro_case_1_validation.cu)
fluid_target_link_libraries(1d_toro_case_1 ${CUDA_LIBRARIES})

# TORO test case 2:
fluid_add_executable(1d_toro_case_2 ${SOURCE_DIR}/1d_toro_case_2_validation.cu)
fluid_target_link_libraries(1d_toro_case_2 ${CUDA_LIBRARIES})

# TORO test case 3:
fluid_add_executable(1d_toro_case_3 ${SOURCE_DIR}/1d_toro_case_3_validation.cu)
fluid_target_link_libraries(1d_toro_case_3 ${CUDA_LIBRARIES})

# TORO test case 4:
fluid_add_executable(1d_toro_case_4 ${SOURCE_DIR}/1d_toro_case_4_validation.cu)
fluid_target_link_libraries(1d_toro_case_4 ${CUDA_LIBRARIES})

# TORO test case 5:
fluid_add_executable(1d_toro_case_5 ${SOURCE_DIR}/1d_toro_case_5_validation.cu)
fluid_target_link_libraries(1d_toro_case_5 ${CUDA_LIBRARIES})

#==--- [1D MM Tests] --------------------------------------------------------==#

# Multimaterial TORO test case 1:
fluid_add_executable(
  1d_mm_toro_case_1 ${SOURCE_DIR}/1d_mm_toro_case_1_validation.cu)
fluid_target_link_libraries(1d_mm_toro_case_1 ${CUDA_LIBRARIES})

# Multimaterial TORO test case 4:
fluid_add_executable(
  1d_mm_toro_case_4 ${SOURCE_DIR}/1d_mm_toro_case_4_validation.cu)
fluid_target_link_libraries(1d_mm_toro_case_4 ${CUDA_LIBRARIES})

# Multimaterial GFM shock tube air-helium test:
fluid_add_executable(
  1d_mm_shock_tube_air_helium ${SOURCE_DIR}/1d_mm_shock_tube_air_helium.cu)
fluid_target_link_libraries(1d_mm_shock_tube_air_helium ${CUDA_LIBRARIES})

# Multimaterial FEDKIW test case 1:
fluid_add_executable(
  1d_mm_fedkiw_test_1 ${SOURCE_DIR}/1d_mm_fedkiw_test_1_validation.cu)
fluid_target_link_libraries(1d_mm_fedkiw_test_1 ${CUDA_LIBRARIES})


#==--- [2D tests] -----------------------------------------------------------==#

# 2D Shock bubble test case:
fluid_add_executable(
  2d_shock_bubble
  ${SOURCE_DIR}/2d_shock_bubble_validation.cu
)
fluid_target_link_libraries(2d_shock_bubble ${CUDA_LIBRARIES})

# 2D Riemann test case 4:
fluid_add_executable(
  2d_liska_case_4
  ${SOURCE_DIR}/2d_liska_case_4_validation.cu
)
fluid_target_link_libraries(2d_liska_case_4 ${CUDA_LIBRARIES})

# 2D Riemann test case 6:
fluid_add_executable(
  2d_liska_case_6
  ${SOURCE_DIR}/2d_liska_case_6_validation.cu
)
fluid_target_link_libraries(2d_liska_case_6 ${CUDA_LIBRARIES})

# 2D Riemann test case 12:
fluid_add_executable(
  2d_liska_case_12
  ${SOURCE_DIR}/2d_liska_case_12_validation.cu
)
fluid_target_link_libraries(2d_liska_case_12 ${CUDA_LIBRARIES})

# 2D Riemann test case 15:
fluid_add_executable(
  2d_liska_case_15
  ${SOURCE_DIR}/2d_liska_case_15_validation.cu
)
fluid_target_link_libraries(2d_liska_case_15 ${CUDA_LIBRARIES})

#==--- [Eikonal tests] ------------------------------------------------------==#

# 2D Eikonal test with speed function: f = 1 and a sphere (circle)
fluid_add_executable(
  2d_eikonal_f_1_sphere
  ${SOURCE_DIR}/2d_eikonal_f_1_sphere.cu
)
fluid_target_link_libraries(2d_eikonal_f_1_sphere ${CUDA_LIBRARIES})

# 2D Eikonal test with speed function: f = 1 and a box (square)
fluid_add_executable(
  2d_eikonal_f_1_box
  ${SOURCE_DIR}/2d_eikonal_f_1_box.cu
)
fluid_target_link_libraries(2d_eikonal_f_1_box ${CUDA_LIBRARIES})

#==--- Finish Up ------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
fluid_create_all_targets()
