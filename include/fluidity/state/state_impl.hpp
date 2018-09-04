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

/// Returns the sum of the velocity squared for each of the velocity conponents
/// for the \p state when the \p state is primitive.
/// \param[in]  state     The state to calculate the squared velocity sum for.
/// \tparam     StateType The type of the state.
template <typename StateType, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto v_squared_sum(StateType&& state)
{
  using state_t = std::decay_t<StateType>;
  auto sum      = static_cast<decltype(state[0])>(0);
  for (const auto i : range(state_t::dimensions))
  {
    const auto& v = state[state_t::index::velocity(i)];
    sum += v * v;
  }
  return sum;
}

/// Returns the sum of the velocity squared for each of the velocity conponents
/// for the \p state when the \p state is conservative.
/// \param[in]  state     The state to calculate the squared velocity sum for.
/// \tparam     StateType The type of the state.
template <typename StateType, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto v_squared_sum(StateType&& state)
{
  using state_t   = std::decay_t<StateType>;
  using indexer_t = typename state_t::index;
  auto sum      = static_cast<decltype(state[0])>(0);
  for (const auto i : range(state_t::dimensions))
  {
    const auto v = state[indexer_t::velocity(i)] / state[indexer_t::density];
    sum += v * v;
  }
  return sum;
}

/// Returns the density of the \p state.
/// \param[in]  state       The state to get the density of.
/// \tparam     StateType   The type of the state.
template <typename StateType>
fluidity_host_device constexpr auto density(StateType&& state)
{
  using state_t   = std::decay_t<StateType>;
  using indexer_t = typename state_t::index;
  return state[indexer_t::density];
}

/// Overload of velocity implementation for pritive form states.
/// \param[in]  state       The state to get the velocity of.
/// \param[in]  dim         The dimension to get the velocity for {x,y,z}.
/// \tparam     StateType   The type of the state.
/// \tparam     Dim         The type of the dimension.
template <typename StateType, typename Dim, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto velocity(StateType&& state, Dim dim)
{
  using state_t   = std::decay_t<StateType>;
  using indexer_t = typename state_t::index;
  return state[indexer_t::velocity(dim)];
}

/// Overload of velocity implementation for conservative form states.
/// \param[in]  state       The state to get the velocity of.
/// \param[in]  dim         The dimension to get the velocity for {x,y,z}.
/// \tparam     StateType   The type of the state.
/// \tparam     Dim         The type of the dimension.
template <typename StateType, typename Dim, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto velocity(StateType&& state, Dim dim)
{
  using state_t   = std::decay_t<StateType>;
  using indexer_t = typename state_t::index;
  return state[indexer_t::velocity(dim)] / state[indexer_t::density];
}

/// Returns the pressure of the \p state, for a primitive form state.
/// \param[in]  state      The state to get the pressure of.
/// \param[in]  mat        The material for the system.
/// \tparam     StateType  The type of the state.
/// \tparam     Mat        The type of the material.
template <typename StateType, typename Mat, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto pressure(StateType&& state, Mat&& mat)
{
  using state_t = std::decay_t<StateType>;
  return state[state_t::index::pressure];
}

/// Returns the pressure of the \p state, for a conservative form state.
/// \param[in]  state      The state to get the pressure of.
/// \param[in]  mat        The material for the system.
/// \tparam     StateType  The type of the state.
/// \tparam     Mat        The type of the material.
template <typename StateType, typename Mat, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto pressure(StateType&& state, Mat&& mat)
{
  using state_t       = std::decay_t<StateType>;
  using value_t       = typename state_t::value_t;  
  using indexer_t     = typename state_t::index;
  constexpr auto one  = value_t{1};
  constexpr auto half = value_t{0.5};
  return (mat.adiabatic() - one)   *
         (state[indexer_t::energy] -
          half * state.density() * state.v_squared_sum());
}

/// Returns the energy for the \p state, when the state is primitive.
/// \param[in]  state     The state to get the energy of.
/// \param[in]  mat       The material for the system.
/// \tparam     StateType The type of the state.
/// \tparam     Mat       The type of the material.
template <typename StateType, typename Mat, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto energy(StateType&& state, Mat&& mat)
{
  using state_t       = std::decay_t<StateType>;
  using value_t       = typename state_t::value_t;
  constexpr auto half = value_t{0.5};
  return state.density() * (half * state.v_squared_sum() + mat.eos(state));
}

/// Returns the energy for the \p state, when the state is conservative.
/// \param[in]  state      The state to get the energy of.
/// \param[in]  material   The material for the system.
/// \tparam     State      The type of the state.
/// \tparam     Material   The type of the material.
template <typename StateType, typename Mat, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto energy(StateType&& state, Mat&&)
{
  using state_t   = std::decay_t<StateType>;
  using indexer_t = typename state_t::index;
  return state[indexer_t::energy];
}

/// Returns the maximum wavespeed for the state, which is defined as:
/// \begin{equation}
///   S_{max} = \max_i \{ |u| + a }
/// \end{equation}
/// where $u_i$ is velocity for this state in dimension $i$, and $a$ is the
/// speed of sound in the \p mat material..
/// \param[in]  state     The state to compute the max wavespeed from.
/// \param[in]  mat       The material for the system.
/// \tparam     StateType The type of the state.
/// \tparam     Mat       The material which describes the system.
template <typename StateType, typename Mat>
fluidity_host_device constexpr auto max_wavespeed(StateType&& state, Mat&& mat)
{
  using state_t     = std::decay_t<StateType>;
  auto max_velocity = std::abs(state.velocity(dim_x));
  for (const auto i : range(state_t::dimensions - 1))
  unrolled_for<state_t::dimensions - 1>([&] (auto i)
  {
    max_velocity = std::max(max_velocity, std::abs(state.velocity(i+1)));
  });
  return max_velocity + mat.sound_speed(state);
}

/// Modifies additional fluxes for a primitive state. When the state is
/// primitive some of the fluxes need to be multiplied by the density, which
/// this function does for a flux at a specific index.
/// \param[in] state      The state vector for the computation.
/// \param[in] flux       The flux vector for the computation.
/// \param[in] i          The index of the flux to set.
/// \tparam    S          The type of the state vector.
/// \tparam    F          The type of the flux vector.
/// \tparam    I          The type of the index.
template <typename S, typename F, typename I, prim_enable_t<S> = 0>
fluidity_host_device constexpr auto 
modify_other_fluxes(S&& state, F&& flux, I i)
{
  flux[i] *= state.density();
}

/// Modifies additional fluxes for conservative state. There is no modification
/// for conservative form states, so this becomes a no-op.
/// \tparam    S        The type of the state vector.
/// \tparam    F        The type of the flux vector.
template <typename S, typename F, typename I, cons_enable_t<S> = 0>
fluidity_host_device constexpr auto modify_other_fluxes(S&&, F&&, I) {}

/// Defines a valid type if the type S has multiple spacial dimensions or if it
/// has additional components.
/// \tparam S The type of the state to base the flux enable on.
template <typename S, typename state_t = std::decay_t<S>>
using other_flux_enable_t =
  std::enable_if_t<(state_t::dimensions            > 1 || 
                    state_t::additional_components > 0), int>;

/// Defines a valid type if the type S does not have multiple spacial
/// dimensions and if it does not have additional components.
/// \tparam S The type of the state to base the flux enable on.
template <typename S, typename state_t = std::decay_t<S>>
using other_flux_disable_t =
  std::enable_if_t<!(state_t::dimensions            > 1 || 
                     state_t::additional_components > 0), int>;

/// Makes additional components of a \p flux container. This overload is
/// enabled if the state has multiple dimensions, or of it has additional
/// components.
/// \param[in] state    The state vector.
/// \param[in] flux     The flux vector.
/// \param[in] dim      The dimension not to compute a flux for.
/// \tparam    S        The type of the state.
/// \tparam    F        The type of the flux container.
/// \tparam    D        The type of the dimension.
template <typename S, typename F, typename D, other_flux_enable_t<S> = 0>
fluidity_host_device constexpr auto
make_other_fluxes(S&& state, F&& flux, D dim)
{
  using state_t        = std::decay_t<S>;
  using indexer_t      = typename state_t::index;
  constexpr auto iters = state_t::dimensions
                       + state_t::additional_components
                       - 1;

  unrolled_for<iters>([&] (auto i)
  {
    const auto idx = indexer_t::v_offset + i + (i >= dim ? 1 : 0);
    flux[idx]      = state[idx] * state.velocity(dim);
    modify_other_fluxes(state, flux, idx);
  });
}

/// Makes additional components of a \p flux container. This overload is
/// enabled if the state does not have multiple dimensions and it does not have
/// additional components.
/// \tparam    S        The type of the state.
/// \tparam    F        The type of the flux container.
/// \tparam    D        The type of the dimension.
template <typename S, typename F, typename D, other_flux_disable_t<S> = 0>
fluidity_host_device constexpr auto make_other_fluxes(S&& s, F&& f, D d) {}

/// Computes the flux for a state. This overload is enabled when the state is
/// primitive.
/// \param[in] state    The state to compute the flux for.
/// \param[in] mat      The material for the system.
/// \param[in] dim      The dimension to compute the flux in terms of.
/// \tparam    S        The type of the state.
/// \tparam    M        The type of the material.
/// \tparam    D        The value of the dimension.
template <typename S, typename M, typename D, prim_enable_t<S> = 0>
fluidity_host_device constexpr auto flux(S&& state, M&& mat, D dim)
{
  using state_t   = std::decay_t<S>;
  using indexer_t = typename state_t::index;
  using storage_t = typename state_t::storage_t;

  const auto& v = state.velocity(dim);

  storage_t flux;
  flux[indexer_t::density]       = state.density() * v;
  flux[indexer_t::pressure]      = v
                                 * (state.energy(mat)
                                 +  state.pressure(mat));
  flux[indexer_t::velocity(dim)] = flux[indexer_t::density] 
                                 * v 
                                 + state.pressure(mat);

  make_other_fluxes(std::forward<S>(state), flux, dim);
  return flux;
}

/// Computes the flux for a state. This overload is enabled when the state is
/// conservative.
/// \param[in] state    The state to compute the flux for.
/// \param[in] mat      The material for the system.
/// \param[in] dim      The dimension to compute the flux in terms of.
/// \tparam    S        The type of the state.
/// \tparam    M        The type of the material.
/// \tparam    D        The value of the dimension.
template <typename S, typename M, typename D, cons_enable_t<S> = 0>
fluidity_host_device constexpr auto flux(S&& state, M&& mat, D dim)
{
  using state_t   = std::decay_t<S>;
  using indexer_t = typename state_t::index;
  using storage_t = typename state_t::storage_t;

  const auto v = state.velocity(dim);
  const auto p = state.pressure(mat);

  storage_t flux;
  flux[indexer_t::density]       = state[indexer_t::velocity(dim)];
  flux[indexer_t::velocity(dim)] = state[indexer_t::velocity(dim)] * v + p;
  flux[indexer_t::energy]        = v * (state.energy(mat) + p);

  make_other_fluxes(std::forward<S>(state), flux, dim);
  return flux;
}

/// Returns the primitive form of the state, regardless of whether the type of
/// the state is primitive or conservative.
/// \param[in] state     The state to get the primitive form of.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template <typename StateType, typename Mat, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto primitive(StateType&& state, Mat&&)
{
  return state;
}

/// Returns a reference to the primitive form of the state, regardless of
/// whether the type of the state is primitive or conservative.
/// \param[in] state     The state to get the primitive form of.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template <typename StateType, typename Mat, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto& primitive(StateType&& state, Mat&&)
{
  return state;
}

/// Returns the primitive form of the state. This overload is enables when the
/// state is conservative.
/// \param[in] state     The state to get the primitive form of.
/// \param[in] mat       The material for the conversion.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template < typename StateType, typename Mat, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto primitive(StateType&& state, Mat&& mat)
{
  using state_t   = std::decay_t<StateType>;
  using result_t  = traits::make_prim_form_t<state_t>;
  using indexer_t = typename result_t::index;
  
  result_t result;
  result[indexer_t::density]  = state.density();
  result[indexer_t::pressure] = state.pressure(mat);

  constexpr auto iters = state_t::dimensions + state_t::additional_components;
  unrolled_for<iters>([&] (auto i)
  {
    const auto index = indexer_t::v_offset + i;
    result[index] = state[index] / state.density();
  });
  return result;
}

/// Returns the conservative form of the state. This overload returns a copy of
/// the state.
/// \param[in] state     The state to get the conservative form of.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template <typename StateType, typename Mat, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto conservative(StateType&& state, Mat&& mat)
{
  return state;
}

/// Returns the conservative form of the state. This overload returns a
/// reference to the state.
/// \param[in] state     The state to get the conservative form of.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template <typename StateType, typename Mat, cons_enable_t<StateType> = 0>
fluidity_host_device constexpr auto& conservative(StateType&& state, Mat&& mat)
{
  return state;
}

// Returns the conservative form of the state. This overload is enables when the
/// state is primitive.
/// \param[in] state     The state to get the primitive form of.
/// \param[in] mat       The material for the conversion.
/// \tparam    StateType The type of the state.
/// \tparam    Mat       The type of the material.
template <typename StateType, typename Mat, prim_enable_t<StateType> = 0>
fluidity_host_device constexpr auto conservative(StateType&& state, Mat&& mat)
{
  using state_t   = std::decay_t<StateType>;
  using result_t  = traits::make_cons_form_t<state_t>;
  using indexer_t = typename result_t::index;

  result_t result;
  result[indexer_t::density] = state.density();
  result[indexer_t::energy]  = state.energy(mat);

  constexpr auto iters = state_t::dimensions + state_t::additional_components;
  unrolled_for<iters>([&] (auto i)
  {
    const auto index = indexer_t::v_offset + i;
    result[index] = state[index] * state.density();
  });
  return result;
}

} // namespace detail
} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_IMPL_HPP
