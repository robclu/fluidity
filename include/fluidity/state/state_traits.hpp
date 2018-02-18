//==--- fluidity/state/state_traits.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_traits.hpp
/// \brief This file defines traits for the state class.
//
//==------------------------------------------------------------------------==//


#ifndef FLUIDITY_STATE_STATE_TRAITS_HPP
#define FLUIDITY_STATE_STATE_TRAITS_HPP

#include <fluidity/container/layout.hpp>

namespace fluid  {

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

namespace state  {

/// Defines the type of the state: primitive or conservative.
enum class Format {
  primitive    = 0,   //!< Stored state data is in primitive form.
  conservative = 1    //!< Stored state data is in conservative form.
};

/// Defines a class to represent a state.
/// \tparam T The type of data used by the state.
template <typename T,
          FormType      Form,         
          std::size_t   Dimensions,
          std::size_t   Components = 0
          StorageFormat Format     = StorageFormat::row_major>
class State;

namespace traits {
namespace detail {

/// Returns true if the type T is a State class. This is the general case for
/// when T is not a State.
/// \tparam T   The type to check for a State type.
template <typename T>
struct IsState {
  /// Returns that the type T is not a State type.
  static constexpr bool value = false;
};

/// Returns true if the type T is a State class. This is the general case for
/// when the class is a State.
/// \tparam T   The type to check for a State type.
template <typename... Ts>
struct IsState<State<Ts...>> {
  static constexpr bool value = true;
};

} // namespace detail

/// Alias for checking if a class is a state. Returns true if the class T is
/// a State class, otherwise returns false.
/// \tparam T The type to check for a State type.
template <typename T>
static constexpr bool is_state_v = detail::IsState<T>::value;

/// Defines the storage type used by the state class. If the storage format is
/// row major, then a traditional array is used which stores each element
/// contigously, otherwise an ArrayView is used and the elements are stored
/// vertically in an SoA manner, for better compute performance.
/// \tparam T   The type of the data to store.
/// \tparam Dimeneions The number of spacial dimensions.
/// \tparam Components The number of additional components, besides, the usual
///         spacial velocities, density, and pressure.
/// \tparam Format     The format to store the the data in.
template <typename T,        
          std::size_t   Dimensions,
          std::size_t   Components = 0
          StorageFormat Format     = StorageFormat::row_major>
using storage_t = 
  std::conditional_t<
    std::is_same_v<Format, StorageFormat::row_major>,
      Array<T, Dimension + Components + 2>,
      ArrayView<T, Dimensions + Components + 2>
  >;

} // namespace traits
} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_TRAITS_HPP