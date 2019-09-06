//==--- fluidity/material/eos_traits.hpp- --------------------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eos_traits.hpp
/// \brief This file defines type traits for equations of state.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_EOS_TRAITS_HPP
#define FLUIDITY_MATERIAL_EOS_TRAITS_HPP

namespace fluid    {
namespace material {

//==--- Forward declarations -----------------------------------------------==//

/// The IdealGas class defines a class for an ideal gas material.
/// \tparam T The type of the data used for computation.
template <typename T> struct IdeadGas;

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
template <typename T> struct StiffFluid;

//==--- [Default traits] ---------------------------------------------------==//

/// The EosTraits class defines traits for an equation of state.
/// \tparam EqnOfState The equation of state to get the traits for.
template <typename EqOfState>
struct EosTraits {};

//== [Specializations]  ----------------------------------------------------==//

/// Specialization of the equation of state traits for an ideal gas.
/// \tparam T The type of the data used by the ideal gas.
template <typename T>
struct EosTraits<IdealGas<T>> {
  /// Defines the data type used by the equation of state.
  using value_t = std::decay_t<T>;
};

/// Specialization of the equation of state traits for a stiff fluid.
/// \tparam T The type of the data used by the stiff fluid.
template <typename T>
struct EosTraits<StiffFluid<T>> {
  /// Defines the data type used by the equation of state.
  using value_t = std::decay_t<T>;
};

}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_EOS_TRAITS_HPP
