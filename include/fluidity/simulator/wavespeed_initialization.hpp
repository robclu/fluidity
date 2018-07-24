//==--- fluidity/simulator/wavespeed_initialization.hpp ---- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  wavespeed_initialization.hpp
/// \brief This file defines the interface for wavespeed initialization.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_WAVESPEED_INITIALIZATION_HPP
#define FLUIDITY_SIMULATOR_WAVESPEED_INITIALIZATION_HPP

#include "wavespeed_initialization.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {
namespace sim   {

/// Sets the wavespeed values pointed to by the \p wavespeeds using the state
/// data pointed to by the \p states iterator. This overload is enabled if the
/// \p states iterator has a CPU execution policy.
/// \param[in] states     Iterator to the state data.
/// \param[in] wavespeeds Iterator to the wavespeed data.
/// \param[in] mat        The material for the system.
/// \tparam    I1         The type of the state iterator.
/// \tparam    I2         The type of the wavespeed iterator.
/// \tparam    Mat        The type of the material.
template <typename I1, typename I2, typename Mat, exec::cpu_enable_t<I1> = 0>
void set_wavespeeds(I1&& states, I2&& wavespeeds, Mat&& mat)
{
  // Call CPU implementation ...
}

/// Sets the wavespeed values pointed to by the \p wavespeeds using the state
/// data pointed to by the \p states iterator. This overload is enabled if the
/// \p states iterator has a GPU execution policy.
/// \param[in] states     Iterator to the state data.
/// \param[in] wavespeeds Iterator to the wavespeed data.
/// \param[in] mat        The material for the system.
/// \tparam    I1         The type of the state iterator.
/// \tparam    I2         The type of the wavespeed iterator.
/// \tparam    Mat        The type of the material.
template <typename I1, typename I2, typename Mat, exec::gpu_enable_t<I1> = 0>
void set_wavespeeds(I1&& states, I2&& wavespeeds, Mat&& mat)
{
  detail::cuda::set_wavespeeds(std::forward<I1>(states)    ,
                               std::forward<I2>(wavespeeds),
                               std::forward<Mat>(mat)     );
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_WAVESPEED_INITIALIZATION_HPP