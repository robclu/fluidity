//==--- fluidity/algorithm/fill.cuh ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fill.cuh
/// \brief This file defines the definitions of the kernels used to fill
///        containers.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FILL_CUH
#define FLUIDITY_ALGORITHM_FILL_CUH

#if defined(__CUDACC__)

#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace detail {
namespace cuda   {

template <typename Iterator, typename P, typename... Args>
fluidity_global void
fill_impl(Iterator begin, P&& pred, Args&&... args)
{
  using it_value_t   = std::decay_t<decltype(*begin)>;
  using pred_value_t = std::decay_t<P>;

  const auto offset = flattened_id(dim_x);
  if constexpr (std::is_same<it_value_t, pred_value_t>::value)
  {
    begin[offset] = pred;
  }
  else
  {
    pred(begin[offset], std::forward<Args>(args)...);
  }
}

/// Fills the range of values defined by { end - begin } using \p pred to set
/// the value of the elements. The \p pred can either be a value or a callable
/// object.
/// \param[in] begin    The iterator to start filling from.
/// \param[in] end      The iterator to end filling at.
/// \param[in] pred     The predicate to use to set the value.
/// \param[in] args     Additional arguments if \p pred is callable.
/// \tparam    Iterator The type of the iterator.
/// \tparam    P        The type of the predicate.
/// \tparam    Args     The type of arguments for a callable predicate.
template <typename Iterator, typename P, typename... Args>
fluidity_global void
fill(Iterator begin, Iterator end, P&& pred, Args&&... args)
{
  const int      elements    = end - begin;
  constexpr auto max_threads = 256;

  dim3 threads_per_block(elements > max_threads ? elements : max_threads);
  dim3 num_blocks(elements / threads_per_block.x);

  fluidity_check_cuda_result(
    fill_impl<<<num_blocks, threads_per_block>>>(begin, pred, args...)
  );
}

}}} // namespace fluid::detail::cuda

#endif // __CUDACC__
#endif // FLUIDITY_ALGORITHM_FILL_CUH