//==--- fluidity/setting/flux_method_option.hpp ------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_method_option.hpp
/// \brief This file defines an option for choosing a flux method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_FLUX_METHOD_OPTION_HPP
#define FLUIDITY_SETTING_FLUX_METHOD_OPTION_HPP

#include "option.hpp"
#include <fluidity/flux_method/flux_methods.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a flux method.
struct FluxMethodOption : Option<FluxMethodOption> {
  /// Defines the type of the choice list for the data options.
  using choice_list_t =
    std::tuple<OptionHolder<flux::Force>        ,
               OptionHolder<flux::Hllc>         ,
               OptionHolder<flux::LaxFriedrichs>,
               OptionHolder<flux::Richtmyer>    >;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = std::tuple_size<choice_list_t>::value;
  /// Defines the type of the option.
  static constexpr const char* type   = "flux_method";

  /// Defines the choices for the option.
  constexpr auto choices() const
  {
    return choice_list_t{"force", "hllc", "lax-friedrichs", "richtmyer"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_LIMIT_FORM_OPTION_HPP