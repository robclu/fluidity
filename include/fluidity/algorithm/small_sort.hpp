//==--- fluidity/algorithm/small_sort.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  small_sort.hpp
/// \brief This file provides fast implementations for sorting a small number
///        of items,
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_SMALL_SORT_HPP
#define FLUIDITY_ALGORITHM_SMALL_SORT_HPP

#include <fluidity/container/array.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {

/// Returns the values, sorted. This implementation is provided so that it's
/// possible to call `small_sort` when unpacking an array.
/// \param[in] a  The first (and only, in this case) value to sort.
/// \tparam    T  The type of the data to sort.
template <typename T>
fluidity_host_device auto small_sort(T a) -> Array<T, 1> {
  return Array<T, 1>{a};
}

/// Returns an array of the sorted input values.
/// \param[in] a  The first value to sort.
/// \param[in] b  The second value to sort.
/// \tparam    T  The type of the data to sort.
template <typename T>
fluidity_host_device auto small_sort(T a, T b) -> Array<T, 2> {
  return Array<T, 2>{std::min(a, b), std::max(a, b)};
}

/// Returns an array of the sorted input values.
/// \param[in] a  The first value to sort.
/// \param[in] b  The second value to sort.
/// \param[in] c  The third value to sort.
/// \tparam    T  The type of the data to sort.
template <typename T>
fluidity_host_device auto small_sort(T a, T b, T c) {
  auto result = Array<T, 3>{std::min(a, b), T(0), std::max(a, b)};
  result[1]   = a + b + c - result[0] - result[2];
  return result;
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_SMALL_SORT_HPP

