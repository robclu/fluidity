//==--- fluidity/material/ideal_gas.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ideal_gas.hpp
/// \brief This file implements functionality for an ideal gas.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_IDEAL_GAS_HPP
#define FLUIDITY_MATERIAL_IDEAL_GAS_HPP

//#include "material.hpp"
#include <fluidity/utility/portability.hpp>
#include <cmath>

namespace fluid    {
namespace material {

/// The IdealGas class defines a class for an ideal gas material.
/// \tparam T The type of the data used for computation.
template <typename T>
struct IdealGas {
  /// Defines the type of the data used by the material.
  using value_t = std::decay_t<T>;

  /// Sets the value of the gas to have the default adiabatic index of 1.4.
  constexpr IdealGas() = default;

  /// Sets the adiabatic index for the gas to have the value of \p adi_index.
  /// \param[in] adi_index The adiabatic index of the gas.
  fluidity_host_device constexpr IdealGas(value_t adi_index)
  : _adi_index(adi_index) {}

  /// Returns the value of the adiabatic index for the ideal gas.
  fluidity_host_device constexpr value_t& adiabatic() noexcept
  {
    return _adi_index;
  }

  /// Returns the value of the adiabatic index for the ideal gas.
  fluidity_host_device constexpr value_t adiabatic() const noexcept
  {
    return _adi_index;
  }

  /// Evaluates the equation of state for the ideal gas, which is given by:
  ///
  ///   \begin{equation}
  ///     e = e(p, \rho) = \frac{p}{(\gamma - 1) \rho}
  ///   \end{equation}
  /// 
  /// and returns the result.
  /// \param[in]  state   The state to use to evaluate the quation of state.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr value_t eos(State&& state) const
  {
    return state.pressure(*this) / 
          ((_adi_index - value_t{1}) * state.density());
  }

  /// Calculates the speed of sound for the ideal gas, based on the equation
  /// of state, where the sound speed is given by:
  ///
  ///   \begin{equation}
  ///     a = \sqrt{\frac{\gamma p}{\rho}}
  ///   \end{equation}
  ///
  /// and returns the result.
  /// \param[in]  state   The state to use to compute the sound speed.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr value_t sound_speed(State&& state) const
  {
    return std::sqrt(_adi_index * state.pressure(*this) / state.density());
  } 

  /// Computes the density for a \p state_to such that it will have the same
  /// entropy as \p state_from for this equation of state. This returns the
  /// density for \p state_to such that setting the density of \p state_to with
  /// the returned density and the given pressure for \p state_to, will result
  /// in \p state_to and \p state_from having the same entropy.
  /// \param[in] state_from
  /// \param[in] state_to

  template <typename StateFrom, typename StateTo>
  fluidity_host_device constexpr value_t
  density_for_const_entropy(const StateFrom& state_from,
                            const StateTo&   state_to  ) const
  {
    const auto entropy = eos(state_from);
    return state_to.pressure(*this) / ((_adi_index - value_t{1}) / entropy);
  }

  /// Computes the density for a \p state_to such that it will have the same
  /// entropy as \p state_from for this equation of state. This returns the
  /// density for \p state_to such that setting the density of \p state_to with
  /// the returned density and the given pressure for \p state_to, will result
  /// in \p state_to and \p state_from having the same entropy.
  /// \param[in] state_from
  /// \param[in] state_to

  template <typename StateFrom, typename StateTo>
  fluidity_host_device constexpr value_t
  density_for_const_entropy_log(const StateFrom& state_from,
                                const StateTo&   state_to  ) const
  {
    const auto a_inv = 1.0 / _adi_index;
    const auto s //= log(state_from.pressure(*this)) 
                 //- _adi_index * log(state_from.density());
                 = log(std::pow(state_from.pressure(*this), a_inv)
                 / state_from.density());
    return std::pow(state_to.pressure(*this), a_inv) * std::exp(s);
    //return (state_to.pressure(*this) - exp(s)) / exp(_adi_index);
  }


 private:
  value_t _adi_index = 1.4; //!< The adiabatic index for the gas.
};

} // namespace material
} // namespace fluid

#endif // FLUIDITY_MATERIAL_IDEAL_GAS_HPP
