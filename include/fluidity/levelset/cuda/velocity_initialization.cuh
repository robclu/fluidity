//==--- fluidity/levelset/cuda/velocity_initialization.cuh - -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  velocity_initialization.cuh
/// \brief This file implements cuda functionality for loading levelset 
///        velocities.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_CUDA_VELOCITY_INITIALIZATION_HPP
#define FLUIDITY_LEVELSET_CUDA_VELOCITY_INITIALIZATION_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace levelset {
namespace cuda     {
namespace detail   {

/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container.
///
/// Sets the value of the velocity pointed to by the \p vel_it iterator, using
/// the state data pointed to by the \p state_it, for the \p dim.
///
/// This overload os only enabled when both iterators have only a single
/// dimension.
///
/// \param[in] vel_it        An iterator to the velocity data to set.
/// \param[in] state_it      An interator to the state data to use to set the
///                          velocity data.
/// \param[in] dim           The dimension to set the velocity for.
/// \tparam    VelIterator   The type of the velocity iterator.
/// \tparam    StateIterator The type of the iterator to the state data.
/// \tparam    Dim           The type of the dimension specifier.
template <
  typename VelIterator  ,
  typename StateIterator,
  typename Dim          ,
  std::enable_if_t<
    (std::decay_t<VelIterator>::dimensions == 1) &&
    (std::decay_t<StateIterator>::dimensions == 1)
    , int
  > = 0
>
fluidity_device_only auto set_velocity_it_value(
  VelIterator&&        vel_it  ,
  const StateIterator& state_it,
  Dim                  dim
) -> void {
  *vel_it = state_it->velocity(dim);
}


/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container.
///
/// Sets the value of the velocity pointed to by the \p vel_it iterator, using
/// the state data pointed to by the \p state_it, for the \p dim.
///
/// This overload os only enabled when both iterators have more than a single
/// dimension.
///
/// \param[in] vel_it        An iterator to the velocity data to set.
/// \param[in] state_it      An interator to the state data to use to set the
///                          velocity data.
/// \param[in] dim           The dimension to set the velocity for.
/// \tparam    VelIterator   The type of the velocity iterator.
/// \tparam    StateIterator The type of the iterator to the state data.
/// \tparam    Dim           The type of the dimension specifier.
template <
  typename VelIterator  ,
  typename StateIterator,
  typename Dim          ,
  std::enable_if_t<
    (std::decay_t<VelIterator>::dimensions > 1) &&
    (std::decay_t<StateIterator>::dimensions > 1),
    int
  > = 0
>
fluidity_device_only auto set_velocity_it_value(
  VelIterator&&        vel_it  ,
  const StateIterator& state_it,
  Dim                  dim
) -> void {
  (*vel_it)[dim] = state_it->velocity(dim);
}

/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container.
///
/// \param[in] mat_iters          Container with iterators for the material
///                               simulation data for each material.
/// \param[in] vel_it             Iterator to the velocity data.
/// \param[in] dh                 The domain resolution to use to compute the
///                               norm.
/// \tparam    MaterialIterators  The type of the container for the material
///                               data iterators. 
/// \tparam    VelIterator        The type of the velocity iterator.
/// \tparam    T                  The data type for the resolution.
template <typename MaterialIterators, typename VelIterator, typename T>
fluidity_global auto set_velocities(
  MaterialIterators mat_iters,
  VelIterator       vel_it   ,
  T                 dh
) -> void {
  constexpr auto dims = std::decay_t<
    decltype(*(get<0>(mat_iters).state_iterator()))
  >::dimensions;
    
  // Offset the iterators to the correct place:
  unrolled_for<dims>([&] (auto dim) {
    const auto flat_offset = flattened_id(dim);
    vel_it.shift(flat_offset, dim);
    for_each(mat_iters, [&] (auto& mat_it) {
      mat_it.shift(flat_offset, dim);
    });
  });

  // Set the velocity using the appropriate levelset:
  bool set = false;
  int idx = 0;
  for_each(mat_iters, [&] (auto& mat_it) {
    idx++;
    if (!set && levelset::inside(mat_it.levelset_iterator())) {
      set = true;
      *vel_it = math::dot(
        mat_it.state_iterator()->velocity_vec(),
        mat_it.levelset_iterator().norm(dh)
      );
      //unrolled_for<dims>([&] (auto dim) {
      //  set_velocity_it_value(vel_it, mat_it.state_iterator(), dim);
      //});
    } 
  });
}

} // namespace detail

/// Sets the velocity values stored in the \p velocities iterator using the
/// state data for each of the \p materials in the materials container. This
/// sets the velocity value as
/// 
/// \begin{equation}
///   v_n = \textbf{v} \dot \textbf{n
/// \end{equation}
/// 
/// where $v$ is a vector of the velocities from state which the material is in,
/// and $n$ is the normal vector for the corresponding cell.
///
/// \param[in] materials          Container with material simulation data for
///                               each material.
/// \param[in] vel_it             Iterator to the velocity data.
/// \param[in] dh                 The grid spatial resolution, required for the
///                               computation of the norm.
/// \tparam    MaterialContainer  The type of the container for the material
///                               data. 
/// \tparam    VelIterator        The type of the velocity iterator.
/// \tparam    T                  The type of the grid resolution.
template <
  typename MaterialContainer,
  typename VelIterator      ,
  typename T
>
auto set_velocities(
  MaterialContainer&& materials,
  VelIterator&&       vel_it   ,
  T                   dh 
) -> void {
  // Get the wrappers which wrap the iterators for each of the materials.
  auto mat_iters = unpack(materials, [&] (auto&&... material) {
    return make_tuple(material.material_iterator()...);
  });

  auto it      = get<0>(mat_iters).state_iterator();
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  detail::set_velocities<<<threads, blocks>>>(mat_iters, vel_it, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::levelset::cuda

#endif // FLUIDITY_LEVELSET_CUDA_VELOCITY_INITIALIZATION_HPP
