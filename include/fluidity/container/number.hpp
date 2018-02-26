//==--- fluidity/container/Number.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  number.hpp
/// \brief This file defines a wrapper for a compile time number.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_NUMBER_HPP
#define FLUIDITY_CONTAINER_NUMBER_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// The Number class wraps an integer so that it can be used at compile time.
/// \tparam Value The value of the number.
template <int Value>
struct Number {
  /// Returns the value of the number.
  static constexpr int value = Value;

  /// Overload of conversion to int operator. Returns the value of the number.
  fluidity_host_device constexpr operator int() const
  {
    return Value;
  }
};

} // namespace fluid

#endif // FLUIDITY_CONTAINER_NUMBER_HPP