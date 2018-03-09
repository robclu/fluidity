//==--- fluidity/algorithm/max_element.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  max_element.hpp
/// \brief This file defines a file defines functionality to compute the
///        maximum element from a container.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_MAX_ELEMENT_HPP
#define FLUIDITY_ALGORITHM_MAX_ELEMENT_HPP

#include <fluidity/iterator/tensor_iterator.hpp>
#include <type_traits>

namespace fluid {

/// Functor to compute the maximum element given two elements.
struct MaxElement {
  /// Overload of operator() to evaluate if the \p left argument is greater than
  /// or equal to the \p right element.
  /// \param[in]  left    The left value for comparison.
  /// \param[in]  right   The right value for comparison.
  /// \tparam     T       The type of the left argument.
  template <typename T>
  fluidity_host_device bool operator()(T&& left, T&& right) const 
  {
    return left >= right;
  }
};

/// Fills the range of values defined by { end - begin } using \p pred to set
/// the value of the elements. The \p pred can either be a value or a callable
/// object.
/// \param[in] begin    The iterator to start filling from.
/// \param[in] end      The iterator to end filling at.
/// \param[in] p        The predicate to use to set the value.
/// \param[in] args     Additional arguments if \p pred is callable.
/// \tparam    Iterator The type of the iterator.
/// \tparam    P        The type of the predicate.
/// \tparam    Args     The type of arguments for a callable predicate.
template <typename Iterator, typename P, typename... Args>
fluidity_host_only auto
max_element(Iterator begin, Iterator end, P&& p = MaxElement(), Args&&... args)
{
  return reduce(begin, end, std::forward<P>(p), std::forward<Args>(args)...);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_MAX_ELEMENT_HPP