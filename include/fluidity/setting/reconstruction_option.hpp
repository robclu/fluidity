//==--- fluidity/setting/reconstruction_option.hpp --------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reconstruction_option.hpp
/// \brief This file defines an option for choosing a reconstructor.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_RECONSTRUCTION_OPTION_HPP
#define FLUIDITY_SETTING_RECONSTRUCTION_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include "option_tuple.hpp"
#include <fluidity/reconstruction/reconstructors.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a data type.
/// \tparam Limiter The type of the limiter for the reconstructor.
template <typename Limiter>
struct ReconOption : Option<ReconOption<Limiter>> {
  /// Defines the type of the choice list.
  using choice_list_t = 
    OptionTuple<OptionHolder<recon::BasicReconstructor<Limiter>>,
                OptionHolder<recon::MHReconstructor<Limiter>>   >;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = choice_list_t::size;
  /// Defines the type of the option.
  static constexpr const char* type   = "reconstructor";

  /// Defines the choices for the option.
  static constexpr auto choice_list()
  {
    return choice_list_t{"none", "muscl-hancock"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_RECONSTRUCTION_OPTION_HPP