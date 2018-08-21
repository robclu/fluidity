//==--- fluidity/setting/dimension_option.hpp -------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dimension_option.hpp
/// \brief This file defines an option for choosing a number of dimensions.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_DIMENSION_OPTION_HPP
#define FLUIDITY_SETTING_DIMENSION_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include <fluidity/utility/number.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing an execution type.
struct DimensionOption : Option<DimensionOption> {

  /// Defines the type of the choice list.
  using choice_list_t = 
    std::tuple<OptionHolder<Num<1>>,
               OptionHolder<Num<2>>,
               OptionHolder<Num<3>>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = std::tuple_size<choice_list_t>::value;
  /// Defines the type of the option.
  static constexpr const char* type   = "dimensions";

  /// Defines the choices for the option.
  constexpr auto choices() const
  {
    return choice_list_t{"one", "two", "three"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_DIMENSION_OPTION_HPP