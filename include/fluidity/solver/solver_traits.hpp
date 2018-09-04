//==--- fluidity/solver/solver_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  solver_traits.hpp
/// \brief This file defines a class which holds the traits of a solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SOLVER_TRAITS_HPP
#define FLUIDITY_SOLVER_SOLVER_TRAITS_HPP

#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace solver {

/// Defines the types of solver implementations available.
enum class Type {
  split   = 0,  //!< Defines a split type implementation.
  unsplit = 1,  //!< Defines an unsplit type implementation.
};

/// The SolverTraits class defines the properties of a solver.
/// \tparam Loader        The type used to load in data for the solver.
/// \tparam Reconstructor The type used to reconstruct cell data.
/// \tparam FluxEvaluator The type used to evaluate the flux between cells.
template <typename Loader, typename Reconstructor, typename FluxEvaluator>
struct SolverTraits {
  /// Defines the type of the data loader for the solver.
  using loader_t         = std::decay_t<Loader>;
  /// Defines the tyoe of the reconstructor for the solver.
  using reconstructor_t  = std::decay_t<Reconstructor>;
  /// Defines the type of the flux evaluator for the solver.
  using flux_evaluator_t = std::decay_t<FluxEvaluator>;
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_SOLVER_TRAITS_HPP