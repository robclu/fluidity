//==--- fluidity/levelset/cuda/levelset_projection.cuh ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_projection.cuh
/// \brief This file defines the implementation which projects a collection of 
///        levelsets such that they do not have any operlaps or voids.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_CUDA_LEVELSET_PROJECTION_CUH
#define FLUIDITY_LEVELSET_CUDA_LEVELSET_PROJECTION_CUH

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace levelset {
namespace cuda     {

/// Defines a valid type if there are 2 levelsets.
/// \tparam Levelsets The container of levelsetts.
template <typename Levelsets>
using two_ls_enable_t = std::enable_if_t<(tuple_size_v<Levelsets> == 2), int>;

/// Defines a valid type if there are more than two levelsets.
/// \tparam Levelsets The container of levelsets.
template <typename Levelsets>
using multi_ls_enable_t = std::enable_if_t<(tuple_size_v<Levelsets> > 2), int>;

/// This performs a projection of the levelsets which ensures that there are no
/// voids or overlaps of the levelsets. The projection is that of:
///
/// [Losasso - Multiple Interacting Fluids - 2006]
/// [http://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf]
///
/// The method is simple, and for each cell, finds the mimimum two levelset
/// values for that cells, computes the average, and then subtracts the average
/// from each of the levelsets for the cells. See paper for more info.
///
/// This function is the specific implementation of the above projection, but is
/// specifically for the case that there are two levelsets. This makes the
/// second levelset the inverse of the first.
/// 
/// \param[in] ls_iterators       A collection of levelset iterators.
/// \tparam    LevelsetIterators  The type of the iterator container. This must
///                               be able to be used with for_each().
template <typename LevelsetIterators, two_ls_enable_t<LevelsetIterators> = 0>
fluidity_global void project_impl(LevelsetIterators ls_iterators)
{
  auto& front_ls = get_front(ls_iterators);
  auto& back_ls  = get_back(ls_iterators);

  unrolled_for<std::decay_t<decltype(back_ls)>::dimensions>([&] (auto dim)
  {
    front_ls.shift(flattened_id(dim), dim);
    back_ls.shift(flattened_id(dim), dim);

    *back_ls = -(*front_ls);
  });
}

/// This performs a projection of the levelsets which ensures that there are no
/// voids or overlaps of the levelsets. The projection is that of:
///
/// [Losasso - Multiple Interacting Fluids - 2006]
/// [http://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf]
///
/// The method is simple, and for each cell, finds the mimimum two levelset
/// values for that cells, computes the average, and then subtracts the average
/// from each of the levelsets for the cells. See paper for more info.
///
/// This function is the specific implementation of the above projection, but is
/// specifically for the case that there are multiple levelsets. When there are
/// only two levelsets, the implementation is simpler and is specialized for
/// that case.
/// 
/// \param[in] ls_iterators       A collection of levelset iterators.
/// \tparam    LevelsetIterators  The type of the iterator container. This must
///                               be able to be used with for_each().
template <typename LevelsetIterators, multi_ls_enable_t<LevelsetIterators> = 0>
fluidity_global void project_impl(LevelsetIterators ls_iterators)
{
  using iter_t        = std::decay_t<decltype(get<0>(ls_iterators))>;
  using value_t       = typename iter_t::value_t;
  constexpr auto dims = iter_t::dimensions;

  // Offset all the iterators and compute the two min values.
  auto smallest      = std::numeric_limits<value_t>::max() - value_t{1};
  auto next_smallest = std::numeric_limits<value_t>::max();
  for_each(ls_iterators, [&] (auto&& it)
  {
    unrolled_for<dims>([&] (auto dim)
    {
      it.shift(flattened_id(dim), dim);
    });
    if (*it < smallest)
    {
      next_smallest = smallest;
      smallest      = *it;
    }
    else if (*it < next_smallest)
    {
      next_smallest = *it;
    }
  });

  // Compute the average which is used to project each cell:
  smallest = value_t{0.5} * (smallest + next_smallest);
  // Perform the projection. Iterators were shifted above
  // so are pointing to the correct cells.
  for_each(ls_iterators, [&] (auto&& it)
  {
    *it -= smallest;
  });
}

/// This performs a projection of the levelsets which ensures that there are no
/// voids or overlaps of the levelsets. The projection is that of:
///
/// [Losasso - Multiple Interacting Fluids - 2006]
/// [http://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf]
///
/// The method is simple, and for each cell, finds the mimimum two levelset
/// values for that cells, computes the average, and then subtracts the average
/// from each of the levelsets for the cells. See paper for more info.
///
/// This simply creates the necessary configuration to launch the implementation
/// kernel.
/// 
/// \param[in] ls_iterators       A collection of levelset iterators.
/// \tparam    LevelsetIterators  The type of the iterator container. This must
///                               be able to be used with for_each().
template <typename LevelsetIterators>
void project(LevelsetIterators&& ls_iterators)
{
  using iter_t        = std::decay_t<decltype(get<0>(ls_iterators))>;
  using value_t       = typename iter_t::value_t;
  constexpr auto dims = iter_t::dimensions;

  auto threads = exec::get_thread_sizes(get<0>(ls_iterators));
  auto blocks  = exec::get_block_sizes(get<0>(ls_iterators), threads);

  project_impl<<<blocks, threads>>>(ls_iterators);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::levelset::cuda


#endif // FLUIDITY_LEVELSET_CUDA_LEVELSET_PROJECTION_HPP
