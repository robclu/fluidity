//==--- fluidity/state/state_component.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_component.hpp
/// \brief This file defines a class which defines a component of a state, with
///        an assosciated name value. It can be used to define state values when
///        initializing a simulatio.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_STATE_COMPONENT_HPP
#define FLUIDITY_STATE_STATE_COMPONENT_HPP

namespace fluid {
namespace state {

/// The StateComponent class is a utility class which can be used to create user
/// defined literals for state components so that components of the state can be
/// set more clearly.
/// \tparam Chars The characters for the name.
template <char... Chars>
struct StateComponent {
  double value = 0.0; //!< Defines the value for the element.
    
  /// Default constructor which is enabled so that we can use decltype on an
  /// empty StateElement to determine the type.
  constexpr StateComponent() = default;

  /// Constructor which sets the value of the component to \p v.
  /// \param[in] v The value for the component.
  constexpr StateComponent(double v) : value(v) {}
    
  /// Returns the name of the component.
  auto name() const
  {
    const char n[sizeof...(Chars) + 1]  = { Chars..., '\0' };
    return std::string(n);
  }
};

/// Utility function which can be used with declval to easily define new state
/// component type aliases, for example:
/// \begin{code}
///   using rho_t = decltype("rho"_component);
/// \end{code}
/// will create a new StateComponent with the name "rho".
/// \tparam Char The type of the name characters
/// \tparam C    The characters in the name.
template <typename Char, Char... C>
constexpr StateComponent<C...> operator "" _component()
{
  return StateComponent<C...>();
}

}} // namespace fluid::state 

#endif // FLUIDITY_STATE_STATE_COMPONENT_HPP