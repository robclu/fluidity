//==--- fluidity/setting/limiter_option.hpp ---------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limiter_option.hpp
/// \brief This file defines an option for choosing the limiter.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_LIMITER_OPTION_HPP
#define FLUIDITY_SETTING_LIMITER_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include "option_tuple.hpp"
#include <fluidity/limiting/limiters.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a limiter.
/// \tparam Form The form of the limiting.
template <typename Form>
struct LimiterOption : Option<LimiterOption<Form>> {
  /// Defines the type of the choice list for the data options.
  using choice_list_t =
    OptionTuple<OptionHolder<limit::Void<Form>>    ,
                OptionHolder<limit::Linear<Form>>  ,
                OptionHolder<limit::VanLeer<Form>> ,
                OptionHolder<limit::Superbee<Form>>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = choice_list_t::size;
  /// Defines the type of the option.
  static constexpr const char* type   = "limiter";

  /// Defines the choices for the option.
  static constexpr auto choice_list()
  {
    return choice_list_t{"void", "linear", "van-leer", "superbee"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_LIMIT_FORM_OPTION_HPP