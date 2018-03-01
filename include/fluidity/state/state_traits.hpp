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

#include <fluidity/container/array.hpp>
#include <fluidity/container/array_view.hpp>
#include <fluidity/container/layout.hpp>
#include <fluidity/dimension/dimension.hpp>

namespace fluid {
namespace state {

/// Defines the type of the state: primitive or conservative.
enum class FormType {
  primitive    = 0,   //!< Stored state data is in primitive form.
  conservative = 1    //!< Stored state data is in conservative form.
};

/// Defines a class to represent a state.
/// \tparam T The type of data used by the state.
template <typename T,
          FormType      Form,         
          std::size_t   Dimensions,
          std::size_t   Components = 0,
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
template <typename T, FormType F, std::size_t D, std::size_t C, StorageFormat S>
struct IsState<State<T, F, D, C, S>> {
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
          std::size_t   Components = 0,
          StorageFormat Format     = StorageFormat::row_major>
using storage_t = 
  std::conditional_t<
    Format == StorageFormat::row_major,
      Array<T, Dimensions + Components + 2>,
      ArrayView<T, Dimensions + Components + 2>
  >;

} // namespace traits
} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_TRAITS_HPP