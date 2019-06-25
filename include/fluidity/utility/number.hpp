//==--- fluidity/utility/number.hpp ------------------------ -*- C++ -*- ---==//
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
/// \brief This file defines a class which represents a compile time constant
///        number type, where a number can be represented as a time, and also
///        have a value at runtime.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_NUMBER_HPP
#define FLUIDITY_UTILITY_NUMBER_HPP

#include "portability.hpp"

namespace fluid {

/// The Num struct stores a number as a type, but also provides functionality to
/// convert the value of the number into a numeric type at runtime or compile
/// time. It is useful in a metaprogramming context to use as a type, but also
/// when the numeric value of the type is required. It works better with
/// variadic templates than using a raw numeric literal.
/// \tparam   Value  The value of the number.
template <std::size_t Value>
struct Num {
  /// Returns the value of the index.
  static constexpr auto value = std::size_t{Value};

  /// Conversion to size_t so that the number can be used as a size type.
  fluidity_host_device constexpr operator size_t()
  {
    return Value;
  }
};

} // namespace fluid

#endif // FLUIDITY_UTILITY_NUMBER_HPP