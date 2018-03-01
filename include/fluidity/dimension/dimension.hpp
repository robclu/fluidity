//==--- fluidity/dimension/dimension.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dimension.hpp
/// \brief This file defines dimension related concepts.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_DIMENSION_DIMENSION_HPP
#define FLUIDITY_DIMENSION_DIMENSION_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// Defines a class to represent a dimension, where the value is known at
/// compile time. The class should be used through the aliases when a single
/// dimension must be specified, i.e:
/// 
/// ~~~cpp
/// // Not clear, what is 0?
/// do_something(container, 0);
/// 
/// // More clear, intention of x dimension application is known at call site.
/// do_something(container, dim_x);
/// ~~~
/// 
/// The other use case is when used with `unrolled_for`, where is can be used
/// more generically:
/// 
/// ~~~cpp
/// unrolled_for<num_dimensions>([] (auto dim_value) {
///   // Compile time dimension can be used here:
///   do_something(container, Dimension<dim_value>{});
/// })
/// ~~~
/// 
/// \tparam Value   The value of the dimension.
template <std::size_t Value>
struct Dimension {
  /// Returns the value of the dimension.
  static constexpr std::size_t value = Value;

  /// Overload of operator size_t to convert a dimension to a size_t.
  fluidity_host_device constexpr operator size_t() const
  {
    return Value;
  }
};

/// Aliases for the x spacial dimension type.
using dimx_t = Dimension<0>;
/// Alias for the y spacial dimension type.
using dimy_t = Dimension<1>;
/// Aloas for the z spacial dimension type.
using dimz_t = Dimension<2>;

/// Defines a compile time type for the x spacial dimension.
static constexpr dimx_t dim_x = dimx_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimy_t dim_y = dimy_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimz_t dim_z = dimz_t{};

} // namespace fluid

#endif // FLUIDITY_DIMENSION_DIMENSION_HPP