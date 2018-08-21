//==--- fluidity/setting/execution_option.hpp -------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_option.hpp
/// \brief This file defines an option for choosing an execution method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_EXECUTION_OPTION_HPP
#define FLUIDITY_SETTING_EXECUTION_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing an execution type.
struct ExecutionOption : Option<ExecutionOption> {

  /// Defines the type of the choice list.
  using choice_list_t = 
    std::tuple<OptionHolder<exec::cpu_type>,
               OptionHolder<exec::gpu_type>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = std::tuple_size<choice_list_t>::value;
  /// Defines the type of the option.
  static constexpr const char* type   = "execution";

  /// Defines the choices for the option.
  constexpr auto choices() const
  {
    return choice_list_t{"cpu", "gpu"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_EXECUTION_OPTION_HPP