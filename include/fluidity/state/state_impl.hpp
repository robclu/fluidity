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
static constexpr const char* const state_concept_error =
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
  using value_t = typename state_t::value_t;
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");

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
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");
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
  using index_t = typename state_t::index;
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");

  if constexpr (state_t::format == FormType::primitive)
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
velocity(State&& state, Dimension<V> /*dim*/) noexcept
{
  using state_t = std::decay_t<State>;
  using index_t = typename state_t::index;
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");

  if constexpr (state_t::format == FormType::primitive)
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
  using value_t = typename state_t::value_t;
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");

  if constexpr (state_t::format == FormType::primitive)
  {
    return state[state_t::index::pressure];
  } 
  else 
  {
    return (material.adiabatic() - value_t{1})
           * (state[state_t::index::energy]
           -  value_t{0.5} * state.density() * state.v_squared_sum());
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
  static_assert(is_state_v<state_t>,
    "Attempt to invoke a state function on a type which is not a state");

  if constexpr (state_t::format == FormType::primitive)
  {
    return state.density() * 
           (0.5 * state.v_squared_sum() + material.eos(state));
  } 
  else 
  {
    return state[state_t::index::energy];
  }
}

/// Makes additional components of the \p flux container.
/// \param[in] state    The state to compute the flux for.
/// \param[in] flux     The flux to compute the values of the additional
///            components for.
/// \param[in] dim      The dimension not to compute a flux for.
/// \tparam    State    The type of the state.
/// \tparam    Flux     The type of the flux container.
/// \tparam    Value    The value which defines the dimension.
template <typename State, typename Flux, std::size_t Value>
fluidity_host_device inline constexpr auto
make_other_fluxes(State&& state, Flux&& flux, Dimension<Value> /*dim*/)
{
  using state_t     = std::decay_t<State>;
  constexpr auto it = state_t::dimensions + state_t::additional_components - 1;

  unrolled_for<it>([&] (auto i)
  {
    constexpr auto dim   = Dimension<Value>{};
    constexpr auto index = state_t::index::v_offset + i + (i >= Value ? 1 : 0);

    flux[index] = state[index] * state.velocity(dim);
    if constexpr (state_t::format == FormType::primitive)
    {
      flux[index] *= state.density();
    }
  });
}

/// Computes the flux for a state.
/// \param[in] state    The state to compute the flux for.
/// \param[in] mat      The material for the system.
/// \param[in] dim      The dimension to compute the flux in terms of.
/// \tparam    State    The type of the state.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value of the dimension.
template <typename State, typename Material, std::size_t Value>
fluidity_host_device inline constexpr auto
flux(State&& state, Material&& mat, Dimension<Value> /*dim*/)
{
  using state_t   = std::decay_t<State>;
  using index_t   = typename state_t::index;
  using storage_t = typename state_t::storage_t;

  constexpr auto dim = Dimension<Value>{};
  const     auto v   = state.velocity(dim);
  const     auto p   = state.pressure(mat);
  const     auto e   = state.energy(mat);

  storage_t flux;
  flux[index_t::density]       = state.density() * v;
  flux[index_t::velocity(dim)] = flux[index_t::density] * v + p;

  if constexpr (state_t::format == FormType::primitive)
  {
    flux[index_t::pressure] = v * (e + p);
  }
  else
  {
    flux[index_t::energy] = v * (e + p);
  }

  if constexpr (state_t::dimensions > 1 || state_t::additional_components > 0)
  {
    make_other_fluxes(std::forward<State>(state), flux, dim);
  }
  return flux;
}

/// Returns the primitive form of the state, regardless of whether the type of
/// the state is primitive or conservative. If the state is conservative, then
/// a conversion is performed to convert the state.
template <typename State, typename Material>
fluidity_host_device inline constexpr auto
primitive(State&& state, Material&& mat)
{
  using state_t  = std::decay_t<State>;
  using result_t = ::fluid::state::State
                      < typename state_t::value_t
                      , FormType::primitive
                      , state_t::dimensions
                      , state_t::additional_components
                      , state_t::storage_layout
                      >;
  using index_t  = typename result_t::index;

  if constexpr (state_t::format == FormType::primitive)
  {
    return state;
  }
  else
  {
    result_t result;
    result[index_t::density]  = state.density();
    result[index_t::pressure] = state.pressure(mat);

    constexpr auto it = state_t::dimensions + state_t::additional_components;
    unrolled_for<it>([&] (auto i)
    {
      constexpr auto index = index_t::v_offset + i;
      result[index] = state[index] / state.density();
    });
    return result;
  }
}

/// Returns the conservative form of the state, regardless of whether the type
/// of the state is primitive or conservative. If the state is primtive, then
/// a conversion is performed to convert the state.
template <typename State, typename Material>
fluidity_host_device inline constexpr auto
conservative(State&& state, Material&& mat)
{
  using state_t  = std::decay_t<State>;
  using result_t = ::fluid::state::State
                      < typename state_t::value_t
                      , FormType::conservative
                      , state_t::dimensions
                      , state_t::additional_components
                      , state_t::storage_layout
                      >;
  using index_t  = typename result_t::index;

  if constexpr (state_t::format == FormType::conservative)
  {
    return state;
  }
  else
  {
    result_t result;
    result[index_t::density] = state.density();
    result[index_t::energy]  = state.energy(mat);

    constexpr auto it = state_t::dimensions + state_t::additional_components;
    unrolled_for<it>([&] (auto i)
    {
      constexpr auto index = index_t::v_offset + i;
      result[index] = state[index] * state.density();
    });
    return result;
  }
}

} // namespace detail
} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_IMPL_HPP
