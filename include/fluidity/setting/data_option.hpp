//==--- fluidity/setting/data_option.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  data_option.hpp
/// \brief This file defines an option for choosing a data type.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_DATA_OPTION_HPP
#define FLUIDITY_SETTING_DATA_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include "option_tuple.hpp"

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a data type.
struct DataOption : Option<DataOption> {
  /// Defines the type of the choice list for the data options.
  using choice_list_t = 
    OptionTuple<OptionHolder<double>, OptionHolder<float>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = choice_list_t::size;
  /// Defines the type of the option.
  static constexpr const char* type   = "data_type";

  /// Defines the choices for the option.
  static constexpr auto choice_list()
  {
    return choice_list_t{"double", "float"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_DATA_OPTION_HPP