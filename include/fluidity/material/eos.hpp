//==--- fluidity/material/eos.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eos.hpp
/// \brief This file defines a material class which is the interface foe
///        equations of state.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_EOS_HPP
#define FLUIDITY_MATERIAL_EOS_HPP

#include "eos_traits.hpp"
#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace material {

/// The EquationOfState class defines an interface to which all equation of
/// state implementations must conform.
/// \tparam EosImpl The implementation of the interface.
template <typename EosImpl>
class EquationOfState {
  /// Defines the type of the reconstructor implementation.
  using impl_t   = EosImpl;
  /// Defines the type of the traits for the reconstructor.
  using traits_t = EosTraits<impl_t>;

  /// Returns a pointer to the implementation.
  fluidity_host_device auto impl() -> implt_* {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device iauto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Defines the data type used by the equation of state.
  using value_t = typename traits_t::value_t;

   /// Returns the value of the adiabatic index for the ideal gas.
  fluidity_host_device constexpr auto adiabatic() noexcept -> value_t& {
    return impl()->adiabatic();
  }

  /// Returns the value of the adiabatic index for the ideal gas.
  fluidity_host_device constexpr auto adiabatic() const noexcept -> value_t {
    return impl()->adiabatic();
  }
  
  /// Evaluates the equation of state for the given \p state.
  /// \param[in]  state   The state to use to evaluate the quation of state.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr auto eos(State&& state) const -> value_t {
    return impl()->eos(std::forward<State>(state));
  }

  /// Calculates the speed of sound for the equation of state for the given \p
  /// state.
  /// \param[in]  state   The state to use to compute the sound speed.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr auto sound_speed(State&& state) const 
  -> value_t {
    return impl()->sound_speed(std::forward<State>(state));
  } 
};

/// Returns true if the type T conforms to the EquaionOfState interface.
/// \tparam T The type to check for conformity to the EquationOfState inteface.
template <typename T>
static constexpr auto is_eos_v = 
  std::is_base_of<EquationOfState<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_EOS_HPP
