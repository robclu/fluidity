//==--- fluidity/setting/parameter/parameter_manager.hpp --- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  parameter_manager.hpp
/// \brief This file defines a class which manages parameters.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_PARAMETER_MANAGER_HPP
#define FLUIDITY_SETTING_PARAMETER_PARAMETER_MANAGER_HPP

#include "parameter_specifier.hpp"
#include <fluidity/algorithm/for_each.hpp>
#include <tuple>

namespace fluid   {
namespace setting {

/// The ParameterManager class defines a manager for parameters which implement
/// the Parameter interface. This essentially allows the form of input file for
/// simulations to be defined. New managers can be defined by simply adding
/// implementations of the paramter interface, and then defining a new manager
/// type. For example, a simulation which only requires an ambient state a shock
/// might use a manager defined as:
///
/// \begin{code}
/// using simple_manager_t = ParameterManager<AmbientParameter, ShockParameter>;
/// \end{code}
///
/// While more complex simulations can take more inputs, such as:
///
/// \begin{code}
/// using complex_manager_t =
///   ParameterManager<
///     AmbientParameter,
///     ShockParameter  ,
///     DomainParameter ,
///     GeometryParameter
///   >;
/// \end{code}
/// \tparam Params The parameter types to manage.
template <typename... Params>
struct ParameterManager {
  /// Defines the type of the container for the parameters.
  using param_container_t = std::tuple<Params...>;

  /// Defines the number of possible parameters.
  static constexpr auto num_params = sizeof...(Params);

  /// Initializes the parameter manager, creating instances of all the required
  /// parameters.
  ParameterManager() {}

  /// Tries to set one of the parameters being managed. Returns true if a
  /// parameter was successfully set.
  /// \param[in] name The name of the paramter to set.
  /// \param[in] seq  The sequence to set a parameter from.
  bool try_set_param(const Setting& setting)
  {
    bool set = false;
    for_each(_params, [&] (auto& param)
    {
      if (param.try_set(setting))
      {
        set = true;
      }
    });
    return set;
  }

  /// Displays the information for all the parameters. This should be called
  /// after all parameters have been set.
  void display_parameters() const
  {
    for_each(_params, [&] (const auto& param)
    {
      std::cout << param.printable_string() << "\n";
    });
  }

  /// Gets a parameter with the name \p param_name.
  /// \param[in] param_name The name of the parameter to get.
  const param_container_t& get_params() const
  {
    return _params;
  }

  /// Checks that all the required paramters are set.
  void check_parameters_are_set()
  {
    for_each(_params, [&] (const auto& param)
    {
      if (param.is_required() && !param.is_set())
      {
        handle_unset_parameter(param);
      }
    });   
  }

 private:
  param_container_t _params; //!< Stores the configurable parameters.

  /// Handles an unset parameter for the case that the parameter is required.
  /// \param[in] spec     The parameter specifier to handle a failure to set.
  /// \tparam    Param    The type of the parameter implementation.
  /// \tparam    Required If the parameter is required.
  template <typename Param, bool Required>
  void handle_unset_parameter(const ParamSpecifier<Param, Required>& spec) const
  {
    if (spec.is_required())
    {
      std::cout << "Failed to set required paramter: " 
                << spec.type() << "\n";
    }
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_HPP