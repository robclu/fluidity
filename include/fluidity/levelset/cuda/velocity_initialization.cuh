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

template <typename VIT, typename MIT>
fluidity_global void set_velocities_impl(VIT velocities, MIT it_wrappers)
{
  constexpr auto dims = std::decay_t<VIT>::dimensions;
    
  // Set the velocity for a cell using the velocity from the material to which
  // the cell belongs.
  unrolled_for<dims>([&] (auto dim)
  {
    bool dim_set = false;
    velocities.shift(flattened_id(dim), dim);

    for_each(it_wrappers, [&] (auto& wrapper)
    {
      wrapper.ls_it.shift(flattened_id(dim), dim);
      wrapper.state_it.shift(flattened_id(dim), dim);
      if (!dim_set && levelset::inside(wrapper.ls_it))
      {
        *velocities = wrapper.state_it->velocity(dim);
        dim_set     = true;
      }
    });
  });
}

template <typename MatData, typename Velocities>
void set_velocities(MatData&& material_data, Velocities&& velocities)
{
  // Get the wrappers which wrap the iterators for each of the materials.
  auto it_wrappers = unpack(material_data, [&] (auto&&... mat_data)
  {
    return make_tuple(mat_data.get_iterator_wrapper()...);
  });

  auto it      = get<0>(it_wrappers).state_it;
  auto threads = exec::get_thread_sizes(it);
  auto blocks  = exec::get_block_sizes(it, threads);

  set_velocities_impl<<<threads, blocks>>>(velocities, it_wrappers);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_LEVELSET_CUDA_VELOCITY_INITIALIZATION_HPP