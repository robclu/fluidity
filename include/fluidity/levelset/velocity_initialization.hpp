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
/// \param[in] dh                 The grid resolution.
/// \tparam    MaterialContainer  The type of the container for the material
///                               data. 
/// \tparam    VelIterator        The type of the velocity iterator.
/// \tparam    T                  The type of the grid resolution.
template <
  typename MaterialContainer,
  typename VelIterator      ,
  typename T                ,
  traits::cpu_enable_t<VelIterator> = 0
>
auto set_velocities(
  MaterialContainer&& materials ,
  VelIterator&&       velocities,
  T                   dh 
) -> void {
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
/// \param[in] dh                 The grid resolution.
/// \tparam    MaterialContainer  The type of the container for the material
///                               data. 
/// \tparam    VelIterator        The type of the velocity iterator.
/// \tparam    T                  The type of the grid resolution.
template <
  typename MaterialContainer,
  typename VelIterator      ,
  typename T                ,
  traits::gpu_enable_t<VelIterator> = 0
>
auto set_velocities(
  MaterialContainer&& materials ,
  VelIterator&&       velocities,
  T                   dh 
)  -> void {
  cuda::set_velocities(
    std::forward<MaterialContainer>(materials),
    std::forward<VelIterator>(velocities)     ,
    dh
  );
}

}} // namespace fluid::levelset

#endif // FLUIDITY_LEVELSET_VELOCITY_INITIALIZATION_HPP
