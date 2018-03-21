//==--- fluidity/algorithm/fill.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fill.hpp
/// \brief This file defines a file which allows a container to be filled.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FILL_HPP
#define FLUIDITY_ALGORITHM_FILL_HPP

#include "fill.cuh"
#include "if_constexpr.hpp"
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <utility>

namespace fluid {

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
template < typename    Iterator
         , typename    T
         , typename... Args
         , std::enable_if_t<
             exec::is_cpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
         >
void fill(Iterator begin, Iterator end, T value, Args&&... args)
{
#if !defined(__CUDACC__)
  constexpr bool not_predicate = 
    std::is_convertible<typename Iterator::value_t, T>::value;

  while (end - begin > 0)
  {
    if constexpr (not_predicate)
    {
      *begin = value;
    }
    else
    {
      value(*begin, std::forward<Args>(args)...);
    }
    ++begin;
  }
#endif // __CUDACC__
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
template < typename Iterator
         , typename T
         , std::enable_if_t<
             exec::is_gpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
         >
void fill(Iterator begin, Iterator end, T value)
{
  detail::cuda::fill(begin, end, value);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FILL_HPP
