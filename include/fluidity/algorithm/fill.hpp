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

#include <fluidity/iterator/tensor_iterator.hpp>
#include <type_traits>

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
template <typename Iterator, typename P, typename... Args>
fluidity_host_only void
fill(Iterator begin, Iterator end, P&& pred, Args&&... args) noexcept
{
  using it_value_t   = std::decay_t<decltype(*begin)>;
  using pred_value_t = std::decay_t<P>;
  while (end - begin > 0)
  {
    if constexpr (std::is_same<it_value_t, pred_value_t>::value)
    {
      *begin = pred;
    }
    else
    {
      pred(begin, std::forward<Args>(args)...);
    }
    ++begin;
  }
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FILL_HPP
