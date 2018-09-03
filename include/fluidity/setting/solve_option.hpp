//==--- fluidity/setting/solve_option.hpp ------------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  solve_option.hpp
/// \brief This file defines an option for choosing a solving method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_SOLVE_OPTION_HPP
#define FLUIDITY_SETTING_SOLVE_OPTION_HPP

#include "option.hpp"
#include "option_holder.hpp"
#include "option_tuple.hpp"
#include <fluidity/solver/solvers.hpp>

namespace fluid   {
namespace setting {

/// Defines an Option implementation for choosing a data type.
/// \tparam Ts The type for for the solver.
template <typename... Ts>
struct SolverOption : Option<SolverOption<Ts...>> {
  /// Defines the type of the choice list.
  using choice_list_t = 
    OptionTuple<OptionHolder<solver::SplitSolver<Ts...>>  ,
                OptionHolder<solver::UnsplitSolver<Ts...>>>;

  /// Defines the number of choices for the option.
  static constexpr size_t num_choices = choice_list_t::size;
  /// Defines the type of the option.
  static constexpr const char* type   = "solve_method";

  /// Defines the choices for the option.
  static constexpr auto choice_list()
  {
    return choice_list_t{"split", "unsplit"};
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_SOLVE_OPTION_HPP