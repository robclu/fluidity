//==--- fluidity/setting/parameter/parameter_managers.hpp -- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  parameter_managers.hpp
/// \brief This file defines types of different paramter managers.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_PARAMETER_MANAGERS_HPP
#define FLUIDITY_SETTING_PARAMETER_PARAMETER_MANAGERS_HPP

#include "ambient_parameter.hpp"
#include "cfl_parameter.hpp"
#include "domain_parameter.hpp"
#include "parameter_manager.hpp"
#include "parameter_specifier.hpp"
#include "shock_parameter.hpp"

namespace fluid   {
namespace setting {

/// Defines a generic parameter manager which requires the following parameters
/// in an input file:
///   - ambient_state : The state in the domain
///   - domain        : Description of the simulation domain
///   - simulator     : Properties of the simulator for the simulation
///   - cfl           : The CFL number for the simulation
///
/// The following additional paramters are optional:
///   - shock         : A shock through the domain
///   - geometry      : Geometry inside the domain
using param_manager_t =
  ParameterManager<
    required_param_t<AmbientParameter>,
    required_param_t<CflParameter>    ,
    required_param_t<DomainParameter> ,
    required_param_t<ShockParameter>
  >;
  
}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_PARAMETER_MANAGERS_HPP