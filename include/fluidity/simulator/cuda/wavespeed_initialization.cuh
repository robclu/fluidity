//==--- fluidity/simulator/cuda/wavespeed_initialization.cuh -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  wavespeed_initialization.cuh
/// \brief This file defines the cuda implemenation for initializin wavespeeds.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_CUDA_WAVESPEED_INITIALIZATION_CUH
#define FLUIDITY_SIMULATOR_CUDA_WAVESPEED_INITIALIZATION_CUH


#include <fluidity/material/material.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <fluidity/dimension/dimension.hpp>

namespace fluid  {
namespace sim    {
namespace detail {
namespace cuda   {

using namespace fluid::material;

/// Sets the wavespeed values in the \p wspeeds iterator using the state
/// data in the \p states iterator.
/// \param[in] states  Iterator to the state data.
/// \param[in] wspeeds Iterator to the wavespeed data.
/// \param[in] mat     The material for the system.
/// \tparam    I1      The type of the state iterator.
/// \tparam    I2      The type of the wavespeed iterator.
/// \tparam    Mat     The type of the material.
template <typename I1 ,
          typename I2 ,
          typename Mat, non_mmaterial_enable_t<Mat> = 1>
fluidity_global void set_wavespeeds_impl(I1 states, I2 wspeeds, Mat mat)
{
#if defined(__CUDACC__)
  const auto idx = flattened_id(dim_x);
  if (idx < wspeeds.size(dim_x))
  {
    wspeeds[idx] = states[idx].max_wavespeed(mat);
  }
#endif
}

/// Sets the wavespeed values in the \p wspeeds iterator using the state
/// data in the \p states iterator.
/// \param[in] states  Iterator to the state data.
/// \param[in] wspeeds Iterator to the wavespeed data.
/// \param[in] mat     The material for the system.
/// \tparam    I1      The type of the state iterator.
/// \tparam    I2      The type of the wavespeed iterator.
/// \tparam    Mat     The type of the material.
template <typename I1 ,
          typename I2 ,
          typename Mat, mmaterial_enable_t<Mat> = 0>
fluidity_global void set_wavespeeds_impl(I1 states, I2 wspeeds, Mat mat)
{
#if defined(__CUDACC__)
  const auto idx = flattened_id(dim_x);
  if (idx < wspeeds.size(dim_x))
  {
    //wspeeds[idx] = states[idx].max_wavespeed(mat);
  }
#endif
}

/// Sets the wavespeed values in the \p wavespeed iterator using the state
/// data in the \p states iterator.
/// \param[in] states     Iterator to the state data.
/// \param[in] wavespeeds Iterator to the wavespeed data.
/// \param[in] mat        The material for the system.
/// \tparam    I1         The type of the state iterator.
/// \tparam    I2         The type of the wavespeed iterator.
/// \tparam    Mat        The type of the material.
template <typename I1, typename I2, typename Mat>
void set_wavespeeds(I1&& states, I2&& wavespeeds, Mat&& mat)
{
#if defined(__CUDACC__)
  // The WS iterator is used to determine the thread and block sizes because the
  // data should be treated as linear (flattened into a single dimension) since
  // the wavespeed computation on a cell is independant of the dimension.
  auto threads = exec::get_thread_sizes(wavespeeds);
  auto blocks  = exec::get_block_sizes(wavespeeds, threads);
  set_wavespeeds_impl<<<blocks, threads>>>(states, wavespeeds, mat);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif
}

/// Sets the wavespeed values in the \p wavespeed iterator using the state
/// data in the \p states iterator.
/// \param[in] states     Iterator to the state data.
/// \param[in] wavespeeds Iterator to the wavespeed data.
/// \param[in] mat        The material for the system.
/// \tparam    I1         The type of the state iterator.
/// \tparam    I2         The type of the wavespeed iterator.
/// \tparam    Mat        The type of the material.
template <typename I1, typename I2, typename Mat>
void set_wavespeeds_mm(I1&& states, I2&& wavespeeds, Mat&& mat)
{
#if defined(__CUDACC__)
  // The WS iterator is used to determine the thread and block sizes because the
  // data should be treated as linear (flattened into a single dimension) since
  // the wavespeed computation on a cell is independant of the dimension.
  auto threads = exec::get_thread_sizes(wavespeeds);
  auto blocks  = exec::get_block_sizes(wavespeeds, threads);
  set_wavespeeds_impl<<<blocks, threads>>>(states, wavespeeds, mat);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif
}

}}}} // namespace fluid::sim::detail::cuda


#endif // FLUIDITY_SIMULATOR_CUDA_WAVESPEED_INITIALIZATION_CUH