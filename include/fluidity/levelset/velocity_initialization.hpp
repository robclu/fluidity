//==--- fluidity/levelset/velocity_initialization.hpp ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  velocity_initialization.hpp
/// \brief This file defines the interface for levelset velocity initialization.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_VELOCITY_INITIALIZATION_HPP
#define FLUIDITY_LEVELSET_VELOCITY_INITIALIZATION_HPP

#include "cuda/velocity_initialization.cuh"
#include <fluidity/traits/device_traits.hpp>

namespace fluid    {
namespace levelset {

/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container.
///
/// This  overload is enabled if the velocity iterator has a GPU execution
/// policy.
///
/// \param[in] materials          Container with material simulation data for
///                               each material.
/// \param[in] velocities         Iterator to the velocity data.
/// \tparam    MaterialContainer  The type of the container for the material
///                               data. 
/// \tparam    VelIterator        The type of the velocity iterator.
template <
  typename MaterialContainer,
  typename VelIterator      ,
  traits::cpu_enable_t<VelIterator> = 0
>
auto set_velocities(MaterialContainer&& materials, VelIterator&& velocities) 
-> void {
  // TODO: Call CPU implementation ...
}

/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container.
///
/// This  overload is enabled if the velocity iterator has a GPU execution
/// policy.
///
/// \param[in] materials          Container with material simulation data for
///                               each material.
/// \param[in] velocities         Iterator to the velocity data.
/// \tparam    MaterialContainer  The type of the container for the material
///                               data. 
/// \tparam    VelIterator        The type of the velocity iterator.
template <
  typename MaterialContainer,
  typename VelIterator      ,
  traits::gpu_enable_t<VelIterator> = 0
>
auto set_velocities(MaterialContainer&& materials, VelIterator&& velocities) 
-> void {
  cuda::set_velocities(
    std::forward<MaterialContainer>(materials),
    std::forward<VelIterator>(velocities)
  );
}

}} // namespace fluid::levelset

#endif // FLUIDITY_LEVELSET_VELOCITY_INITIALIZATION_HPP
