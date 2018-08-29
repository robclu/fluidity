//==--- fluidity/setting/material_option.hpp --------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_option.hpp
/// \brief This file defines an option for choosing a material type.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_MATERIAL_OPTION_HPP
#define FLUIDITY_SETTING_MATERIAL_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include <fluidity/material/materials.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a data type.
/// \tparam T The data type for the material.
template <typename T>
struct MaterialOption : Option<MaterialOption<T>> {
  /// Defines the type of the choice list for the data options.
  using choice_list_t = 
    std::tuple<OptionHolder<material::IdealGas<T>>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = std::tuple_size<choice_list_t>::value;
  /// Defines the type of the option.
  static constexpr const char* type   = "material";

  /// Defines the choices for the option.
  constexpr auto choice_list() const
  {
    return choice_list_t{"ideal-gas"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_DATA_OPTION_HPP