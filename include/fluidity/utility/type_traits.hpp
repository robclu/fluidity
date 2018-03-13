//==--- fluidity/utility/type_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  type_traits.hpp
/// \brief This file defines c++17 wrappers so that they can be used with c++14
///        for CUDA code.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_TYPE_TRAITS_HPP
#define FLUIDITY_UTILITY_TYPE_TRAITS_HPP

#include <type_traits>

namespace fluid {

/// Wrapper for std::is_same to allow is_same_v to work with c++14.
/// \tparam A The first type for comparison.
/// \tparam B The second type for comparison.
template <typename A, typename B>
static constexpr bool is_same_v = std::is_same<A, B>::value;

/// Wrapper for std::decay to allow decay_t to work with c++14.
/// \tparam T The type to decay.
template <typename T>
using decay_t = typename std::decay<T>::type;

} // namespace fluid

#endif // FLUIDITY_UTILITY_TYPE_TRAITS_HPP