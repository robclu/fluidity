//==--- fluidity/ghost_fluid/extrapolate_ghost_cells.hpp --- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  extrapolate_ghost_cells.hpp
/// \brief This file implements the interface for extrapolating ghost cells.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_EXTRAPOLATE_GHOST_CELLS_HPP
#define FLUIDITY_GHOST_FLUID_EXTRAPOLATE_GHOST_CELLS_HPP

#include "cuda/extrapolate_ghost_cells.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {
namespace ghost {

/// Function which extrapolates the ghost cells for a the material. 
/// \param[in]  extrap    The extrapolator to use.
/// \param[in]  mat_iter  The iterator for the material.
/// \param[in]  dh        The resolution of the data.
/// \param[in]  bandw     The width of the band of cells to set.
/// \tparam     E         The type of the extrapolator.
/// \tparam     MIt       The type of the material iterator.
/// \tparam     T         The type of the resolution data.
template <typename E, typename MIt, typename T, exec::cpu_enable_t<MIt> = 0>
void extrapolate_ghost_cells(E&& extrap, MIt&& mat_iter, T dh, std::size_t bandw)
{
}

/// Function which extrapolates the ghost cells for a the material. 
/// \param[in]  extrap    The extrapolator to use.
/// \param[in]  mat_iter  The iterator for the material.
/// \param[in]  dh        The resolution of the data.
/// \param[in]  bandw     The width of the band of cells to set.
/// \tparam     E         The type of the extrapolator.
/// \tparam     MIt       The type of the material iterator.
/// \tparam     T         The type of the resolution data.
template <typename E, typename MIt, typename T, exec::cpu_enable_t<MIt> = 0>
void extrapolate_ghost_cells(E&& extrap, MIt&& mat_iter, T dh, std::size_t bandw)
{
  cuda::extrapolate_ghost_cells(
    std::forward<E>(extrap),
    std::forward<MIt>(mat_iter),
    dh,
    bandw
  );
}

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_EXTRAPOLATE_GHOST_CELLS_HPP
