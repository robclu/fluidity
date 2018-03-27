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

namespace fluid  {
namespace detail {

/// Struct which implements filling based on whether or not a predicate is used.
/// This implementation is for when a predicate is not used.
/// \tparam UsesPredicate If a predicate is used. 
template <bool UsesPredicate = false>
struct FillImpl {
  /// Implements the filling, setting the iterator to the value.
  /// \param[in] it       The iterator to set the value for.
  /// \param[in] value    The value to set the iterator to.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    T        The type of the value.
  template <typename Iterator, typename T>
  static constexpr void apply(Iterator&& it, T value)
  {
    *it = value;
  }
};

template <>
struct FillImpl<true> {
  /// Implements the filling, setting the iterator using the predicate.
  /// \param[in] it       The iterator to set the value for.
  /// \param[in] pred     The predicate to use to set the iterator.
  /// \param[in] args     Additional arguments for the predicate.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Pred     The type of the predicate.
  /// \tparam    Args     The types of additional predicate arguments.
  template <typename Iterator, typename Pred, typename... Args>
  static constexpr void apply(Iterator&& it, Pred&& pred, Args&&... args)
  {
    pred(*it, std::forward<Args>(args)...);
  }
};

} // namespace detail

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
  constexpr bool is_predicate = 
    !std::is_convertible<typename Iterator::value_t, T>::value;

  while (end - begin > 0)
  {
    detail::FillImpl<is_predicate>::apply(begin                      ,
                                          std::forward<T>(value)     ,
                                          std::forward<Args>(args)...);
    ++begin;
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
