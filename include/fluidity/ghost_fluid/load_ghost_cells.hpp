//==--- fluidity/ghost_fluid/load_ghost_cells.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_ghost_cells.hpp
/// \brief This file implements the interface for loading ghost cells.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_LOAD_GHOST_CELLS_HPP
#define FLUIDITY_GHOST_FLUID_LOAD_GHOST_CELLS_HPP

#include "cuda/load_ghost_cells.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {
namespace ghost {

/// Function which loads the ghost cells for each of the materials which are
/// stored in the \p materials container, using the \p gfm method to load the
/// ghost cells.
/// 
/// This overload is enabled if the execution type defined by the material data
/// containers in the \p materials is for the CPU.
///
/// \param[in]  gfm               The ghost fluid method to use.
/// \param[in]  materials         The data for all the materials.
/// \param[in]  dh                The resolution of the data.
/// \tparam     GhostMethod       The type of the ghost fluid method to use.
/// \tparam     MaterialContainer The type of the material data container.
/// \tparam     T                 The type of the resolution data.
template <
  typename GhostMethod      ,
  typename MaterialContainer,
  typename T                ,
  traits::cpu_enable_t<MaterialContainer> = 0
>
auto load_ghost_cells(
  GhostMethod&&       gfm      ,
  MaterialContainer&& materials,
  T                   dh
) -> void {
  // TODO: Add implementation ...
}

/// Function which loads the ghost cells for each of the materials which are
/// stored in the \p materials container, using the \p gfm method to load the
/// ghost cells.
/// 
/// This overload is enabled if the execution type defined by the material data
/// containers in the \p materials is for the GPU.
///
/// \param[in]  gfm               The ghost fluid method to use.
/// \param[in]  materials         The data for all the materials.
/// \param[in]  dh                The resolution of the data.
/// \tparam     GhostMethod       The type of the ghost fluid method to use.
/// \tparam     MaterialContainer The type of the material data container.
/// \tparam     T                 The type of the resolution data.
template <
  typename GhostMethod      ,
  typename MaterialContainer,
  typename T                ,
  traits::gpu_enable_t<MaterialContainer> = 0
>
auto load_ghost_cells(
  GhostMethod&&       gfm      ,
  MaterialContainer&& materials,
  T                   dh
) -> void {
  cuda::load_ghost_cells(
    std::forward<GhostMethod>(gfm)            ,
    std::forward<MaterialContainer>(materials),
    dh                                  
  );
}

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_LOAD_GHOST_CELLS_HPP

