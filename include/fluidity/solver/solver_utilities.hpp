//==--- fluidity/solver/simulation_updater.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.hpp
/// \brief This file defines a class which updates a simulation, and the
///        implementation is specialized for CPU and GPU execution policies.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SOLVER_UTILITIES_HPP
#define FLUIDITY_SOLVER_SOLVER_UTILITIES_HPP

namespace fluid  {
namespace solver {

/// Defines the types of input for a solver input.
enum class Input {
  backward = 0, //!< Defines an input backward of the cell.
  forward  = 1  //!< Defines an input forward of the cell.
};

/// Wrapper class which can be used to overload functions based on the type of
/// the input.
/// \tparam Direction The direction to specialize in terms of.
template <Input Direction>
struct InputSelector {};

/// Alias for creating a backward input state.
using back_input_t = InputSelector<Input::backward>;
/// Aliad for creating a forward input state.
using fwrd_input_t = InputSelector<Input::forward>;

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_SOLVER_UTILITIES_HPP