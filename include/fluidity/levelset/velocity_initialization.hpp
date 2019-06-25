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
#include <fluidity/execution/execution_policy.hpp>

namespace fluid    {
namespace levelset {

/// Sets the velocity values pointed to by the \p velocities using the state
/// data for each of the materials in the material iterator wrapper. This 
/// overload is enabled if the velocity iterator has a CPU execution policy.
/// \param[in] mat_data   Iterator to the material data.
/// \param[in] velocities Iterator to the velocity data.
/// \tparam    MatIt      The type of the state iterator.
/// \tparam    VelIt      The type of the wavespeed iterator.
template <typename MatIt, typename VelIt, exec::cpu_enable_t<VelIt> = 0>
void set_velocities(MatIt&& mat_data, VelIt&& velocities)
{
  // Call CPU implementation ...
}

/// Sets the velocity values pointed to by the \p velocities using the state
/// data for each of the materials in the material iterator wrapper. This 
/// overload is enabled if the velocity iterator has a GPU execution policy.
/// \param[in] mat_data   Iterator to the material data.
/// \param[in] velocities Iterator to the velocity data.
/// \tparam    MatIt      The type of the state iterator.
/// \tparam    VelIt      The type of the wavespeed iterator.
template <typename MatData, typename VelIt, exec::gpu_enable_t<VelIt> = 0>
void set_velocities(MatData&& mat_data, VelIt&& velocities)
{
  cuda::set_velocities(std::forward<MatData>(mat_data),
                       std::forward<VelIt>(velocities));
}

}} // namespace fluid::levelset

#endif // FLUIDITY_LEVELSET_VELOCITY_INITIALIZATION_HPP