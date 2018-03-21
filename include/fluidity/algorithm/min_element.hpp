//==--- fluidity/algorithm/min_element.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  min_element.hpp
/// \brief This file defines functionality to compute the minimum element.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_MIN_ELEMENT_HPP
#define FLUIDITY_ALGORITHM_MIN_ELEMENT_HPP

#include "reduce.hpp"

namespace fluid {

/// Returns the manimum element between the \p begin and \p end elements.
/// \param[in] begin    The first element to find the min element from.
/// \param[in] end      The last element to find the min element from.
/// \tparam    Iterator The type of the iterator.
template <typename Iterator>
fluidity_host_device decltype(auto)
min_element(Iterator&& begin, Iterator&& end)
{
  return 
    reduce(
      std::forward<Iterator>(begin),
      std::forward<Iterator>(end)  ,
      [] fluidity_host_device (auto& a, const auto& b) { a = std::min(a, b); }
    );
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_MIN_ELEMENT_HPP