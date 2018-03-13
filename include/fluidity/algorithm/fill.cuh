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

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace detail {
namespace cuda   {

template <bool True>
struct IfConstexpr;

struct IfConstexpr<true> {
  template <typename P1, typename P2>
  fluidity_host_only decltype(auto)
  operator()(P1&& p1, P2&& p2) const
  {
    p1();
  }
};

struct IfConstexpr<false> {
  template <typename P1, typename P2>
  fluidity_host_only decltype(auto)
  operator()(P1&& p1, P2&& p2) const
  {
    p2();
  }
};

template <bool Condition, typename P1, typename P2>
fluidity_host_device inline constexpr decltype(auto)
if_constexpr(P1&& p1, P2&& p2)
{
  IfConstexpr<Condition>()(std::forward<P1>(p1), std::forward<P2>(p2));
}

template <typename Iterator, typename P, typename... Args>
fluidity_device_only void fill_impl(Iterator begin, P&& pred, Args&&... args)
{
  using it_value_t   = std::decay_t<decltype(*begin)>;
  using pred_value_t = std::decay_t<P>;

  const auto offset = flattened_id(dim_x);
  if_constexpr<is_same_v<it_value_t, pred_value_t>
  (
    [&]
    {
      begin[offset] = pred;
    },
    [&]
    {
      pred(begin[offset], std::forward<Args>(args)...);
    }
  );
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
fill(Iterator begin, Iterator end, P pred, Args... args)
{
  const int      elements    = end - begin;
  constexpr auto max_threads = 256;

  dim3 threads_per_block(elements > max_threads ? elements : max_threads);
  dim3 num_blocks(elements / threads_per_block.x);

  fill_impl<<<num_blocks, threads_per_block>>>(begin, pred, args...);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::detail::cuda

#endif // __CUDACC__
#endif // FLUIDITY_ALGORITHM_FILL_CUH