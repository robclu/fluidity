//==--- fluidity/ghost_fluid/cuda/load_ghost_cells.cuh ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_ghost_cells.cuh
/// \brief This file implements cuda functionality for loading ghost cell data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_CUDA_LOAD_GHOST_CELLS_HPP
#define FLUIDITY_GHOST_FLUID_CUDA_LOAD_GHOST_CELLS_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {
namespace cuda  {

template <typename GFM, typename MatItWrappers, typename T>
fluidity_global void
load_ghost_cells_impl(GFM gfm, MatItWrappers it_wrappers, T dh)
{
  GFM::invoke(it_wrappers, dh);
}

template <typename GFM, typename MatData, typename T>
void load_ghost_cells(GFM&& gfm, MatData&& material_data, T dh)
{
  // Get the wrappers which wrap the iterators for each of the materials.
  auto it_wrappers = unpack(material_data, [&] (auto&&... mat_data)
  {
    return make_tuple(mat_data.get_iterator_wrapper()...);
  });

  auto it      = get<0>(it_wrappers).state_it;
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  load_ghost_cells_impl<<<threads, blocks>>>(gfm, it_wrappers, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_GHOST_FLUID_CUDA_LOAD_GHOST_CELLS_HPP
