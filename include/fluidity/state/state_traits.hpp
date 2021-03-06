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

/// Defines a class to represent a state, which stores density, pressure,
/// velocity components for each spatial dimension, as well as additional
/// components.
/// \tparam T          The type of data used by the state.
/// \tparam Form       The form of the state, i.e primitive, conservative etc.
/// \tparam Dimensions The number of spatial dimensions for the state.
/// \tparam Components The number of additional components for the state.
/// \tparam Format     The storage format for the state data.
template <
  typename      T             ,
  FormType      Form          ,
  std::size_t   Dimensions    ,
  std::size_t   Components = 0,
  StorageFormat Format     = StorageFormat::row_major
>
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
template <
  typename      T         ,
  FormType      Form      ,
  std::size_t   Dimensions,
  std::size_t   Components,
  StorageFormat Storage
>
struct IsState<State<T, Form, Dimensions, Components, Storage>> {
  /// Returns that this is a state.
  static constexpr bool value = true;
};

/// Defines a class which can be used for tag dispatch based on the form of the
/// state.
template <FormType Form> struct StateDispatchTag {};

} // namespace detail

/// Alias for checking if a class is a state. Returns true if the class T is
/// a State class, otherwise returns false.
/// \tparam T The type to check for a State type.
template <typename T>
static constexpr bool is_state_v = detail::IsState<T>::value;

/// Constexpr trait function which returns true if the State is has a primitive
/// form. This can be used to enable function overloads for a primitve state.
/// \tparam State The state to check if in primitive form.
template <typename StateType>
static constexpr auto is_primitive_v = 
  std::decay_t<StateType>::format == FormType::primitive;

/// Constexpr trait function which returns true if the State is has a
/// conservative form. This can be used to enable function overloads for a
/// conservative state.
/// \tparam State The state to check if in primitive form.
template <typename StateType>
static constexpr auto is_conservative_v = 
  std::decay_t<StateType>::format == FormType::conservative;

/// Alias for a primtiive dispatch tag type.
using primitive_tag_t    = detail::StateDispatchTag<FormType::primitive>;
/// Alias for a conservative dispatch tag type;
using conservative_tag_t = detail::StateDispatchTag<FormType::conservative>;

/// Creates a consetexpr instance of a dispatch tag from a state.
/// \tparam State The state to create a dispatch tag for.
template <typename StateType>
static constexpr auto state_dispatch_tag = 
  detail::StateDispatchTag<std::decay_t<StateType>::format>{};

/// Defines a type which enables functions for any state type.
/// \tparam State The type to base the enabling on.
template <typename T>
using state_enable_t = std::enable_if_t<is_state_v<T>, int>;

/// Defines a type which enables functions for any type which is not a state.
/// \tparam State The type to base the enabling on.
template <typename T>
using non_state_enable_t = std::enable_if_t<!is_state_v<T>, int>;

/// Defines a type which enables functions for a primitive state type.
/// \tparam State The type to base the primitive enabling on.
template <typename StateType>
using prim_enable_t = std::enable_if_t<is_primitive_v<StateType>, int>;

/// Defines a type which enables functions for a conservative state type.
/// \tparam State The type to base the conservative enabling on.
template <typename StateType>
using cons_enable_t =
  std::enable_if_t<is_conservative_v<StateType>, int>;

/// Defines a primitive state with the same properties as the S type.
/// \tparam S The state to get a primitive form of.
template <typename S, typename state_t = std::decay_t<S>>
using make_prim_form_t =
  State<
    typename state_t::value_t     ,
    FormType::primitive           ,
    state_t::dimensions           ,
    state_t::additional_components,
    state_t::storage_layout       >;

/// Defines a conservatice state with the same properties as the S type.
/// \tparam S The state to get a conservative form of.
template <typename S, typename state_t = std::decay_t<S>>
using make_cons_form_t =
  State<
    typename state_t::value_t     ,
    FormType::conservative        ,
    state_t::dimensions           ,
    state_t::additional_components,
    state_t::storage_layout       >;

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
