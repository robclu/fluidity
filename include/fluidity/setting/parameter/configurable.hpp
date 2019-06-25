//==--- fluidity/setting/parameter/configurable.hpp -------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  configurable.hpp
/// \brief This file defines a static base class interface for configuring a
//         class using parameters.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_CONFIGURABLE_HPP
#define FLUIDITY_SETTING_PARAMETER_CONFIGURABLE_HPP

#include "parameter.hpp"

namespace fluid   {
namespace setting {

/// The Configurable class defines a static interface for configuring an object
/// using specializations of the Parameter class. To be configurable, a class
/// simply needs to implement overloads of the `configure_impl()˜ method for
/// example, to create a class for which the CFl and resolution can be
/// configured, the following might be done:
/// \begin{code}
/// struct ExampleConfig : public Configurable<ExampleConfig> {
///   // Overload for setting the CFL.
///   void configure_impl(CflParameter cfl_param)
///   {
///     // Set the CFL number ...
///   }
///
///   // Overload for setting the resolution.
///   void configure_impl(ResolutionParameter res)
///   {
///     // Set the resolution ...
///   }
/// };
/// \end{code}
/// \tparam Impl The implementation of the confirable interface.
template <typename Impl>
struct Configurable {
 private:
  /// Defines the implementation type of the configurable interface.
  using impl_t = Impl;
 public:

  /// Forwards the \p param to the implementation for configuration.
  /// \param[in] param     The paramter to use to configure the object.
  /// \tparam    ParamImpl The implementation of the Parameter interface.
  template <typename ParamImpl>
  void configure(const Parameter<ParamImpl>& param)
  {
    impl()->configure_impl(param);
  }
 
 private:
  /// Returns a pointer to a non-const implementation.
  impl_t* impl()
  {
    return static_cast<impl_t*>(this);
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_CONFIGURABLE_HPP