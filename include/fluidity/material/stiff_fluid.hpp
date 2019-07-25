//==--- fluidity/material/stiff_fluid.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  stiff_fluid.hpp
/// \brief This file implements functionality for a fluid with a stiffened
///        equation of state.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_STIFF_FLUID_HPP
#define FLUIDITY_MATERIAL_STIFF_FLUID_HPP

#include <fluidity/utility/portability.hpp>
#include <cmath>

namespace fluid    {
namespace material {

/// The StiffFluid class defines a class for a fluid with a stiffened equation
/// of state, which is defined as:
///
/// \begin{equation}
///   P = \rho e ( \gamma - 1 ) - \gamma P_{\inf}
/// \end{equation}
///
/// where
///   - $\gamma$   = specific heat ratio / Gruneisen exponent
///   - $P_{\inf}$ = material dependant constant
///
/// \tparam T The type of the data used for computation.
template <typename T>
struct StiffFluid {
  /// Defines the type of the data used by the material.
  using value_t = std::decay_t<T>;

  /// Sets the value of the gas to have the default adiabatic index of 1.4.
  constexpr StiffFluid() = default;

  /// Sets the adiabatic index for the fluid to have the value of \p adi_index.
  /// \param[in] adi_index The adiabatic index of the fluid.
  fluidity_host_device constexpr StiffFluid(value_t adi_index)
  : _adi_index(adi_index) {}

  /// Sets the adiabatic index for the fluid to have the value of \p adi_index,
  /// and the material contant $P_{\inf}$ to \p p_inf.
  /// \param[in] adi_index The adiabatic index of the fluid.
  /// \param[in] p_inf     The constant for the fluid.
  fluidity_host_device constexpr StiffFluid(value_t adi_index, value_t p_inf)
  : _adi_index(adi_index), _p_inf(p_inf) {}

  /// Returns the value of the adiabatic index for the fluid.
  fluidity_host_device constexpr value_t& adiabatic() noexcept {
    return _adi_index;
  }

  /// Returns the value of the adiabatic index for the fluid.
  fluidity_host_device constexpr value_t adiabatic() const noexcept {
    return _adi_index;
  }

  /// Evaluates the equation of state for the fluid, which is given by:
  ///
  /// \begin{equation}
  ///   e = e(p, \rho) = \frac{p + \gamma P_{\inf}}{(\gamma - 1) \rho}
  /// \end{equation}
  /// 
  /// and returns the result.
  /// \param[in]  state   The state to use to evaluate the quation of state.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr value_t eos(State&& state) const {
    return (state.pressure(*this) + _adi_index * _p_inf) / 
          ((_adi_index - value_t{1}) * state.density());
  }

  /// Calculates the speed of sound for the fluid, based on the equation
  /// of state, where the sound speed is given by:
  ///
  /// \begin{equation}
  ///   a = \sqrt{\frac{\gamma p}{\rho}}
  /// \end{equation}
  ///
  /// and returns the result.
  /// \param[in]  state   The state to use to compute the sound speed.
  /// \tparam     State   The type of the state.
  template <typename State>
  fluidity_host_device constexpr value_t sound_speed(State&& state) const {
    return std::sqrt(
      _adi_index * (state.pressure(*this) + _p_inf) / state.density()
    );
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
                            const StateTo&   state_to  ) const {
    // TODO: Implement ...
    return value_t{0};
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
                                const StateTo&   state_to  ) const {
    return value_t{0};
  }


 private:
  value_t _adi_index = 5.5;   //!< The adiabatic index for the fluid.
  value_t _p_inf     = 0.613; //!< The material dependant constant.
};


}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_STIFF_FLUID_HPP
