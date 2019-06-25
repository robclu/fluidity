//==--- fluidity/setting/limit_form_option.hpp ------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limit_form_option.hpp
/// \brief This file defines an option for choosing the limiting form.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_LIMIT_FORM_OPTION_HPP
#define FLUIDITY_SETTING_LIMIT_FORM_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include "option_tuple.hpp"
#include <fluidity/limiting/limiter_traits.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a data type.
struct LimitFormOption : Option<LimitFormOption> {
  /// Defines the type of the choice list for the data options.
  using choice_list_t =
    OptionTuple<OptionHolder<limit::cons_form_t>,
                OptionHolder<limit::prim_form_t>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = choice_list_t::size;
  /// Defines the type of the option.
  static constexpr const char* type   = "limit_form";

  /// Defines the choices for the option.
  static constexpr auto choice_list()
  {
    return choice_list_t{"conservative", "primitive"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_LIMIT_FORM_OPTION_HPP