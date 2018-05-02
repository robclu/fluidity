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
/// \param[in] begin The first element to find the max element from.
/// \param[in] end   The last element to find the max element from.
/// \tparam    It    The type of the iterator.
template <typename It>
auto max_element(It&& begin, It&& end)
{
  using value_t = typename It::value_t;

  // Because of cuda only supporting c++14, this can't be a lambda, so we need
  // to create a separate functor for the max predicate ...
  auto max = [] fluidity_host_device (value_t& a, const value_t& b)
  {
    a = std::max(a, b);
  };

  return reduce(std::forward<It>(begin), std::forward<It>(end), max);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_MAX_ELEMENT_HPP