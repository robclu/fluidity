//==--- fluidity/algorithm/for_each.cuh -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  for_each.cuh
/// \brief This file defines the implementation of the cuda version of fill
///        functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FOR_EACH_CUH
#define FLUIDITY_ALGORITHM_FOR_EACH_CUH

#include <fluidity/container/array.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <algorithm>
#include <utility>

namespace fluid  {
namespace detail {
namespace cuda   {

/// Kernel implementation which fills an iterator element \p begin using the
/// provided callable object \p predicate.
/// \param[in] begin     The first iterator to fill with the \p value.
/// \param[in] predicate The value to fill the elements with.
/// \param[in] args      The arguments for the predicate.
/// \tparam    It        The type of the iterator.
/// \tparam    P         The type of the predicate.
/// \tparam    As        The arguments for the predicate.
template <typename It, typename P, multiit_enable_t<It> = 0, typename... As>
fluidity_global void for_each_impl_multi(It it, P pred, As... args)
{
  using it_t = std::decay_t<It>;
  if (in_range(it))
  {
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      it.shift(flattened_id(dim), dim);
    });
    pred(it, args...); 
  }
}

/// Fills the range of values defined by { end - begin } using \p pred to set
/// the value of the elements. The \p pred can either be a value or a callable
/// object.
/// \param[in] begin      The iterator to start filling from.
/// \param[in] end        The iterator to end filling at.
/// \param[in] value_pred The value/predicate to use to set the value.
/// \param[in] args       Additional arguments if \p pred is callable.
/// \tparam    It         The type of the iterator.
/// \tparam    T          The type of the predicate/value.
/// \tparam    Args       The type of arguments for a callable predicate.
template <typename It, typename Pred, typename... Args>
void for_each(It it, Pred&& pred, Args&&... args)
{
#if defined(__CUDACC__)
  auto thread_sizes = get_thread_sizes(it);
  auto block_sizes  = get_block_sizes(it, thread_sizes);

  for_each_impl_multi<<<block_sizes, thread_sizes>>>(it, pred, args...);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

}}} // namespace fluid::detail::cuda

#endif // FLUIDITY_ALGORITHM_FILL_CUH