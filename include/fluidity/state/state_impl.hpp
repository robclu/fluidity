//==--- fluidity/state/state.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_impl.hpp
/// \brief This file defines the implementation of state based functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_STATE_IMPL_HPP
#define FLUIDITY_STATE_STATE_IMPL_HPP

#include "state_traits.hpp"
#include <fluidity/algorithm/unrolled_for.hpp>

namespace fluid  {
namespace state  {
namespace detail {

using namespace traits;

/// Defines a message to print when there is a concept error on a function call,
/// i.e, when an attempt is made to invoke a function where the template param
/// is a State but the argument is not.
static constexpr const char* state_concept_error =
  "Attempt to invoke a state function on a type which is not a state";

/// Returns the sum of the square velocity of each of the velocity conponents
/// for a state.
/// \param[in]  state The state to calculate the squared velocity sum for.
/// \tparam     State The type of the state.
template <typename State>
fluidity_host_device inline constexpr auto
v_squared_sum(State&& state) noexcept
{
  using state_t = std::decay_t<State>;
  using value_t = state_t::value_t;
  static_assert(is_state_v<state_t>, detail::state_concept_error);

  value_t sum = 0;
  unrolled_for<state_t::dimensions>([&sum, &state] (auto i) 
  {
    const auto& v = state[state_t::index::velocity(i)];
    sum += v * v;
  });
  return sum;
}

/// Returns the density of the \p state. There is a compiler error if the
/// \p state is not a State type.
/// \param[in]  state  The state to get the density of.
/// \tparam     State  The type of the state.
template <typename State>
fluidity_host_device inline constexpr auto density(State&& state) noexcept
{
  using state_t = std::decay_t<State>;
  static_assert(is_state_v<state_t>, detail::state_concept_error);
  return state[state_t::index::density];
}

/// Overload of velocity implementation for pritive form states.
/// \param[in]  state   The state to get the velocity of.
/// \param[in]  dim     The dimension to get the velocity for {x,y,z}.
/// \tparam     State   The type of the state.
template <typename State>
fluidity_host_device inline constexpr auto 
velocity(State&& state, std::size_t dim) noexcept
{
  using state_t = std::decay_t<State>;
  using index_t = state_t::index;
  static_assert(is_state_v<state_t>, detail::state_concept_error);

  if constexpr (state_t::format_t == Format::primitive)
  {
    return state[index_t::velocity(dim)];
  } 
  else 
  {
    return state[index_t::velocity(dim)] / state[index_t::density];
  }
}

/// Overload of velocity implementation for pritive form states.
/// \param[in]  state   The state to get the velocity of.
/// \param[in]  dim     The dimension to get the velocity for {x,y,z}.
/// \tparam     State   The type of the state.
template <typename State, std::size_t V>
fluidity_host_device inline constexpr auto 
velocity(State&& state, Dimension<V> dim) noexcept
{
  using state_t = std::decay_t<State>;
  using index_t = state_t::index;
  static_assert(is_state_v<state_t>, detail::state_concept_error);

  if constexpr (state_t::format_t == Format::primitive)
  {
    return state[index_t::velocity(Dimension<V>{})];
  } 
  else 
  {
    return state[index_t::velocity(Dimension<V>{})] / state[index_t::density];
  }
}

/// Returns the pressure of the \p state. There is a compiler error if the
/// \p state is not a state type.
/// \param[in]  state      The state to get the pressure of.
/// \param[in]  material   The material for the system.
/// \tparam     State      The type of the state.
/// \tparam     Material   The type of the material.
template <typename State, typename Material>
fluidity_host_device inline constexpr auto
pressure(State&& state, Material&&  material) noexcept
{
  using state_t = std::decay_t<State>;
  static_assert(is_state_v<state_t>, detail::state_concept_error);

  if constexpr (state_t::format_t == Format::primitive)
  {
    return state[state_t::index::pressure];
  } 
  else 
  {
    return (material.adiabatic() - 1)     *
           (state[state_t::index::energy] -
            0.5 * state.density() * state.v_squared_sum());
  }
}

/// Returns the energy for the \p state.
/// \param[in]  state      The state to get the energy of.
/// \param[in]  material   The material for the system.
/// \tparam     State      The type of the state.
/// \tparam     Material   The type of the material.
template <typename State, typename Material>
fluidity_host_device inline constexpr auto
energy(State&& state, Material&& material) noexcept
{
  using state_t = std::decay_t<State>;
  static_assert(is_state_v<state_t>, detail::state_concept_error);

  if constexpr (state_t::format_t == Format::primitive)
  {
    return state.density() * 
           (0.5 * state.v_squared_sum() + material.eos(state));
  } 
  else 
  {
    return state[state_t::index::energy];
  }
}

} // namespace detail
} // namespace state
} // namespace fluid

#endif FLUIDITY_STATE_STATE_IMPL_HPP
