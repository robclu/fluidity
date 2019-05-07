//==--- fluidity/material/cuda/combine_materials.cuh ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  combine_materials.cuh
/// \brief This file combines the data for multiple materials into a single
///        material such that the data for the single material is valid
///        throughout the domain.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_CUDA_COMBINE_MATERIALS_HPP
#define FLUIDITY_MATERIAL_CUDA_COMBINE_MATERIALS_HPP

#include <fluidity/container/tuple.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/levelset/levelset.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace material {
namespace cuda     {

/// Combines the data from the material \p iterators into the first iterator in
/// the \p iterators container such that the first iterator state data is
/// correct for the entire domain.
///
/// \TODO: This should be modified so that the first material state and levelset
///        data is in shared memory.
/// 
/// \param[in] iterators          A container of material iterators.
/// \tparam    MaterialIterators  The type of material iterator container.
template <typename MaterialIterators>
fluidity_global void combine_into_first_impl(MaterialIterators iterators)
{
  using iter_container_t = std::decay_t<MaterialIterators>;
  for_each(iterators, [&] (auto& mat_iterator)
  {
    using mat_iter_t = std::decay_t<decltype(mat_iterator)>;
    unrolled_for<mat_iter_t::dimensions>([&] (auto dim)
    {
      mat_iterator.shift(flattened_id(dim), dim);
    });
  });

  // The first material iterator is the one that will store correct data.
  auto& final_material_iter = get_front(iterators);
  unrolled_for<tuple_size_v<iter_container_t> - 1>([&] (auto i)
  {
    auto& current_material_iter = get<i + 1>(iterators);

    // NOTE: Here it is assumed that the levelset are all signed distance
    //       functions and that there are no holes or overlaps in the domain.
    if (levelset::inside(current_material_iter.levelset_iterator()))
    {
      final_material_iter.state() = current_material_iter.state();
    }
  });
}

/// Combines the data from the material \p iterators into the first iterator in
/// the \p iterators container such that the first iterator state data is
/// correct for the entire domain.
/// \param[in] iterators          A container of material iterators.
/// \tparam    MaterialIterators  The type of material iterator container.
template <typename MaterialIterators>
void combine_into_first(MaterialIterators&& iterators)
{
  using it_t   = std::decay_t<decltype(get_front(iterators))>;
  auto threads = get_thread_sizes(get_front(iterators));
  auto blocks  = exec::get_block_sizes(get_front(iterators), threads);

  combine_into_first_impl<<<blocks, threads>>>(iterators);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::material::cuda

#endif // FLUIDITY_MATERIAL_CUDA_COMBINE_MATERIALS_HPP
