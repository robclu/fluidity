//==--- fluidity/levelset/cuda/levelset_reinitialization.cuh -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_reinitialization.cuh
/// \brief This file implements cuda functionality for invoking levelset
///        reinitialization. 
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_CUDA_LEVELSET_REINITIALIZATION_HPP
#define FLUIDITY_LEVELSET_CUDA_LEVELSET_REINITIALIZATION_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace levelset {
namespace cuda     {

template <typename Init, typename Levelset, typename Conv>
fluidity_global
void reinit_levelset_init_impl(Init init, Levelset levelset, Conv conv)
{
  using init_t        = std::decay_t<Init>;
  using levelset_t    = std::decay_t<Levelset>;
  constexpr auto dims = levelset_t::dimensions;

  /// Load the levelset data into shared memory:
  unrolled_for<dims>([&] (auto dim)
  {
    *ls.shift(dim, flattened_id(dim)) = *levelset.shift(dim, flattened_id(dim));
    conv.shift(dim, flattened_id(dim));
  });





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

template <typename Init, typename Levelset>
void reinit_levelset(Init&& init, Levelset&& levelset)
{
  using levelset_t = std::decay_t<Levelset>;

  auto threads = exec::get_thread_sizes(levelset);
  auto blocks  = exec::get_block_sizes(levelset, threads);

  // Create boolean data to represent the convergence of each of the nodes.
  auto conv_data = DeviceTensor<bool, levelset_t::dimensions>();
  unrolled_for<levelset_t::dimensions>([&] (auto dim)
  {
    conv_data.resize_dim(dim, levelset.size(dim));
  });
  auto conv_iter = conv_data.multi_iterator();

  reinit_levelset_impl<<<threads, blocks>>>(init, levelset, conv_data);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::ghost::cuda

#endif // FLUIDITY_LEVELSET_CUDA_LEVELSET_REINITIALIZATION_HPP