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

#include "reduce.hpp"

namespace fluid {



/// Returns the maximum element between the \p begin and \p end elements.
/// \param[in] begin    The first element to find the max element from.
/// \param[in] end      The last element to find the max element from.
/// \tparam    Iterator The type of the iterator.
template <typename Iterator>
typename Iterator::value_t max_element(Iterator&& begin, Iterator&& end)
{
  using value_t = typename Iterator::value_t;

  // Because of cuda only supporting c++14, this can't be a lambda, so we need
  // to create a separate functor for the max predicate ...
  struct Max {
    fluidity_host_device void operator()(value_t& a, const value_t& b) const
    {
      a = std::max(a, b);
    }
  };

  return
    reduce(std::forward<Iterator>(begin), std::forward<Iterator>(end), Max{});
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_MAX_ELEMENT_HPP