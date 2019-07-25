//==--- fl../ghost_fluid/cuda/extrapolate_ghost_cells.cuh -- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  extrapolate_ghost_cells.cuh
/// \brief This file implements cuda functionality for extrapolating ghost cell
///        data using cuda.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_HPP
#define FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {
namespace cuda  {

template <typename E, typename MIt, typename T>
fluidity_global void
extrapolate_ghost_cells_impl(E e, MIt mat_iter, T dh, std::size_t bandw)
{
  E::invoke(it_wrappers, dh);
}

template <typename MatData, typename E, typename T>
fluidity_global void
extrapolate_ghost_cells(MatData&& mat_data, E&& extrap, T dh, std::size_t bandw)
{
  // Get the wrappers which wrap the iterators for each of the materials.
  auto mat_iters = unpack(mat_data, [&] (auto&&... m_data)
  {
    return make_tuple(m_data.material_iterator()...);
  });

  auto it      = get<0>(it_wrappers).state_it;
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  for_each(mat_iters, [&] (auto& mi) {
    extrapolate_ghost_cells_impl<<<threads, blocks>>>(extrap, mi, dh, bandw);
    fluidity_check_cuda_result(cudaDeviceSynchronize());
  });
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_HPP
