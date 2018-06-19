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
/// \brief This file defines the implementation of the cuda version of fill
///        functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FILL_CUH
#define FLUIDITY_ALGORITHM_FILL_CUH

#include "if_constexpr.hpp"
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <algorithm>
#include <utility>

namespace fluid  {
namespace detail {
namespace cuda   {

/// Kernel implementation which fills an iterator element \p begin with the
/// value defined by \p value.
/// \param[in] begin    The first iterator to fill with the \p value.
/// \param[in] value    The value to fill the elements with.
/// \tparam    Iterator The type of the iterator.
/// \tparam    T        The type of the element the iterator holds.
template < typename Iterator
         , typename T
         , enable_if_t<
             std::is_convertible<T, typename Iterator::value_t>::value, int
           > = 0
         >
fluidity_global void fill_impl(Iterator begin, T value)
{
  begin[flattened_id(dim_x)] = value;
}

/// Kernel implementation which fills an iterator element \p begin using the
/// provided callable object \p predicate.
/// \param[in] begin     The first iterator to fill with the \p value.
/// \param[in] predicate The value to fill the elements with.
/// \param[in] args      The arguments for the predicate.
/// \tparam    Iterator  The type of the iterator.
/// \tparam    Pred      The type of the predicate.
/// \tparam    Args      The arguments for the predicate.
template < typename    Iterator
         , typename    Pred
         , enable_if_t<
            !std::is_convertible<Pred, typename Iterator::value_t>::value, int
           > = 0
         , typename... Args
         >
fluidity_global void fill_impl(Iterator begin, Pred pred, Args... args)
{
  pred(begin[flattened_id(dim_x)], args...);
}

/// Fills the range of values defined by { end - begin } using \p pred to set
/// the value of the elements. The \p pred can either be a value or a callable
/// object.
/// \param[in] begin      The iterator to start filling from.
/// \param[in] end        The iterator to end filling at.
/// \param[in] value_pred The value/predicate to use to set the value.
/// \param[in] args       Additional arguments if \p pred is callable.
/// \tparam    Iterator   The type of the iterator.
/// \tparam    T          The type of the predicate/value.
/// \tparam    Args       The type of arguments for a callable predicate.
template <typename Iterator, typename T, typename... Args>
void fill(Iterator begin, Iterator end, T&& value_pred, Args&&... args)
{
#if defined(__CUDACC__)
  const int      elements    = end - begin;
  constexpr auto max_threads = 256;

  dim3 threads_per_block(elements < max_threads ? elements : max_threads);
  dim3 num_blocks(std::max(elements / threads_per_block.x,
                           static_cast<unsigned int>(1)));

  fill_impl<<<num_blocks, threads_per_block>>>(begin, value_pred, args...);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

}}} // namespace fluid::detail::cuda

#endif // FLUIDITY_ALGORITHM_FILL_CUH