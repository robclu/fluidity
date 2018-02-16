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

namespace fluid {

/// Fills the range of values defined by { end - begin } with the value of
/// \p value.
/// \param[in] begin    The iterator to start filling from.
/// \param[in] end      The iterator to end filling at.
/// \param[in] value    The value to set the pointed to elements to.
/// \tparam    Iterator The type of the iterator.
/// \tparam    T        The type of the data to fill the elements with.
template <typename Iterator, typename T>
fluidity_host_only void fill(Iterator begin, Iterator end, T value) noexcept
{
  while (end - begin > 0) {
    *begin = value;
    ++begin;
  }
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FILL_HPP
