//==--- fluidity/levelset/levelset_projection.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_projection.hpp
/// \brief This file defines the interface for projecting a collection of
///        levelsets such that they all agree on the interface locations and
///        have no overlaps or voids.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_LEVELSET_PROJECTION_HPP
#define FLUIDITY_LEVELSET_LEVELSET_PROJECTION_HPP

#include "cuda/levelset_projection.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid    {
namespace levelset {


/// This performs a projection of the levelsets which ensures that there are no
/// voids or overlaps of the levelsets. The projection is that of:
///
/// [Losasso - Multiple Interacting Fluids - 2006]
/// [http://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf]
///
/// The method is simple, and for each cell, finds the mimimum two levelset
/// values for that cells, computes the average, and then subtracts the average
/// from each of the levelsets for the cells.
/// 
/// This ensures the following properties of the levelsets after the
/// projection:
///
/// - The smallest element, $\phi_j$, is the only negative element and is the
///   distance to the interface. This removes overlaps since it is the *only*
///   negative element, and removes voids since there is a negative element.
///
/// - If the above holds and $\phi_k$ is the second smallest element, then
///   $\phi_k = - \phi_j$. This makes all $\phi \neq \phi_j$ positive and signed
///   distance functions.
///
/// This overload is only enabled which the iterators have a CPU execution
/// policy. This will cause a compiler error of the execution policy of the
/// iterators are not all the same.
/// 
/// \param[in] ls_iterators       A collection of levelset iterators.
/// \tparam    LevelsetIterators  The type of the iterator container. This must
///                               be able to be used with for_each().
template <typename LevelsetIterators, exec::cpu_enable_t<LevelsetIterators> = 0>
void project(LevelsetIterators&& ls_iterators)
{
}

/// This performs a projection of the levelsets which ensures that there are no
/// voids or overlaps of the levelsets. The projection is that of:
///
/// [Losasso - Multiple Interacting Fluids - 2006]
/// [http://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf]
///
/// The method is simple, and for each cell, finds the mimimum two levelset
/// values for that cells, computes the average, and then subtracts the average
/// from each of the levelsets for the cells.
/// 
/// This ensures the following properties of the levelsets after the
/// projection:
///
/// - The smallest element, $\phi_j$, is the only negative element and is the
///   distance to the interface. This removes overlaps since it is the *only*
///   negative element, and removes voids since there is a negative element.
///
/// - If the above holds and $\phi_k$ is the second smallest element, then
///   $\phi_k = - \phi_j$. This makes all $\phi \neq \phi_j$ positive and signed
///   distance functions.
///
/// This overload is only enabled which the iterators have a GPU execution
/// policy. This will cause a compiler error of the execution policy of the
/// iterators are not all the same.
/// 
/// \param[in] ls_iterators       A collection of levelset iterators.
/// \tparam    LevelsetIterators  The type of the iterator container. This must
///                               be able to be used with for_each().
template <typename LevelsetIterators, exec::gpu_enable_t<LevelsetIterators> = 0>
void project(LevelsetIterators&& ls_iterators)
{
  cuda::project(std::forward<LevelsetIterators>(ls_iterators));
}

}} // namespace fluid::levelset


#endif // FLUIDITY_LEVELSET_LEVELSET_PROJECTION_HPP
