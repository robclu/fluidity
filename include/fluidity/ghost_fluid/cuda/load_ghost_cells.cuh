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

namespace fluid  {
namespace ghost  {
namespace cuda   {
namespace detail {

/// Function which loads the ghost cells for each of the materials which are
/// iterated over using the iterators in the \p mat_iters container, using
/// the \p gfm method to load the ghost cells.
/// 
/// \param[in]  gfm               The ghost fluid method to use.
/// \param[in]  mat_iters         The material iterators.
/// \param[in]  dh                The resolution of the data.
/// \tparam     GhostMethod       The type of the ghost fluid method to use.
/// \tparam     MaterialIterators The type of the material iterator container.
/// \tparam     T                 The type of the resolution data.
template <typename GhostMethod, typename MaterialIterators, typename T>
fluidity_global auto load_ghost_cells(
  GhostMethod       gfm      ,
  MaterialIterators mat_iters, 
  T                 dh
) -> void {
  GhostMethod::invoke(mat_iters, dh);
}

} // namespace detail

/// Function which loads the ghost cells for each of the materials which are
/// iterated over using the iterators in the \p mat_iters container, using
/// the \p gfm method to load the ghost cells.
/// 
/// \param[in]  gfm               The ghost fluid method to use.
/// \param[in]  mat_iters         The material iterators.
/// \param[in]  dh                The resolution of the data.
/// \tparam     GhostMethod       The type of the ghost fluid method to use.
/// \tparam     MaterialIterators The type of the material iterator container.
/// \tparam     T                 The type of the resolution data.
template <typename GhostMethod, typename MaterialContainer, typename T>
auto load_ghost_cells(GhostMethod&& gfm, MaterialContainer&& materials, T dh)
-> void {
  // Get the wrappers which wrap the iterators for each of the materials.
  auto mat_iters = unpack(materials, [&] (auto&&... material) {
    return make_tuple(material.material_iterator()...);
  });

  auto it      = get<0>(mat_iters).state_iterator();
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  detail::load_ghost_cells<<<threads, blocks>>>(gfm, mat_iters, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_GHOST_FLUID_CUDA_LOAD_GHOST_CELLS_HPP
