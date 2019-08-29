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

#ifndef FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_CUH
#define FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_CUH

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {
namespace cuda  {

/// Invokes the extrapolation of the star state in the \p mat_iter using the
/// levelset of the \p mat_iter and the \p extrapolator, for a witdth of \p
/// extrap_width cells.
/// \param[in] extrapolator The extrapolator to use.
/// \param[in] mat_iter     The material iterator for the data.
/// \param[in] dh           The resolution of the domain.
/// \param[in] extrap_width The number of cells to extrapolate the star state
///                         to.
/// \tparam    Extrapolator The type of the extrapolator.
/// \tparam    MatIterator  The type of the material iterator.
/// \tparam    T            The type of the resolution
template <typename Extrapolator, typename MatIterator, typename T>
fluidity_global auto extrapolate_star_state_impl(
  Extrapolator extrapolator,
  MatIterator  mat_iter    ,
  T            dh          ,
  std::size_t  extrap_width
) -> void {
  Extrapolator::invoke(mat_iter, dh, extrap_width);
}

/// Extrapolates the star state in the \p mat_iter state iterator using the
/// levelset of the \p mat_iter and the \p extrapolator, for a witdth of \p
/// extrap_width cells.
///
/// The star state is found by finding the first cell inside the material from
/// the interface.
///
/// \param[in] extrapolator The extrapolator to use.
/// \param[in] mat_iter     The material iterator for the data.
/// \param[in] dh           The resolution of the domain.
/// \param[in] extrap_width The number of cells to extrapolate the star state
///                         to.
/// \tparam    Extrapolator The type of the extrapolator.
/// \tparam    MatIterator  The type of the material iterator.
/// \tparam    T            The type of the resolution
template <typename Extrapolator, typename MaterialData, typename T>
fluidity_host_only auto extrapolate_star_state(
  Extrapolator   extrapolator,
  MaterialData&& mat_data    ,
  T              dh          ,
  std::size_t    extrap_width
) -> void {
  // Make the iterators for each of the materials ...
  auto mat_iters = unpack(mat_data, [&] (auto&&... m_data) {
    return make_tuple(m_data.material_iterator()...);
  });

  // Here it is assumed that all the data is the same size, which it always
  // should be for Eularian based methods.
  auto it      = get<0>(mat_iters).state_iterator();
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  // Extrapolate each of the materials.
  for_each(mat_iters, [&] fluidity_host_only (auto& m_it) {
    extrapolate_star_state_impl<<<threads, blocks>>>(
      extrapolator,
      m_it        ,
      dh          ,
      extrap_width
    );
    fluidity_check_cuda_result(cudaDeviceSynchronize());
  });
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_GHOST_FLUID_CUDA_EXTRAPOLATE_GHOST_CELLS_CUH
