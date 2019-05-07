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

/// Function which loads the ghost cells for each of the materials 
/// \param[in]  gfm            The ghost fluid method to use.
/// \param[in]  material_data  The data for all the materials.
/// \param[in]  dh             The resolution of the data.
/// \tparam     GFM            The type of the ghost fluid method to use.
/// \tparam     MatData        The type of the material data.
/// \tparam     T              The type of the resolution data.
template <typename GFM    ,
          typename MatData,
          typename T      , exec::cpu_enable_t<MatData> = 0>
void load_ghost_cells(GFM&& gfm, MatData&& material_data, T dh)
{
}

/// Function which loads the ghost cells for each of the materials which are
/// stored in the \p material_data. This overload is enabled if the execution
/// type defined in the \p material_data is for the GPU.
/// \param[in]  gfm            The ghost fluid method to use.
/// \param[in]  material_data  The data for all the materials.
/// \param[in]  dh             The resolution of the data.
/// \tparam     GFM            The type of the ghost fluid method to use.
/// \tparam     MatData        The type of the material data.
/// \tparam     T              The type of the resolution data.
template <typename GFM    ,
          typename MatData,
          typename T      , exec::gpu_enable_t<MatData> = 0>
void load_ghost_cells(GFM&& gfm, MatData&& material_data, T dh)
{
  cuda::load_ghost_cells(std::forward<GFM>(gfm)              ,
                         std::forward<MatData>(material_data),
                         dh                                  );
}

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_LOAD_GHOST_CELLS_HPP

