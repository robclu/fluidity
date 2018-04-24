//==--- fluidity/simulator/simulation_traits.hpp ---------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  siulation_traits.hpp
/// \brief This file defines traits of a simulation.
//
//==------------------------------------------------------------------------==//

#include <fluidity/solver/boundary_loader.hpp>
#include <fluidity/solver/split_solver.hpp>
#include <fluidity/solver/unsplit_solver.hpp>
#include <fluidity/solver/solver_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace sim   {

/// The Simulation Traits class defines the properties of a simulation.
template < typename     State
         , typename     Material
         , typename     Reconstructor
         , typename     FluxSolver
         , solver::Type SolverImpl
         , typename     ExecutionPolicy = exec::default_type
         >
struct SimulationTraits {
 private:
  /// Defines the traits of the solver for the simulation.
  
 public:
  /// Defines the type of the state used for the simulation.
  using state_t         = std::decay_t<State>;
  /// Defines the type of material used for the simulation.
  using material_t      = std::decay_t<Material>;
  /// Defines the type of the reconstructor used for the simulation.
  using reconstructor_t = std::decay_t<Reconstructor>;
  /// Defines the type of the solver used for the simulation.
  using flux_solver_t   = std::decay_t<FluxSolver>;
  /// Defines the type of the boundary loader.
  using loader_t        = solver::BoundaryLoader<reconstructor_t::width>;
  /// Defines the type of the execution policty for the simulation.
  using execution_t     = ExecutionPolicy;
  /// Defines the type of the data used by the state.
  using value_t         = typename state_t::value_t;

  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto spacial_dims     = state_t::dimensions;
  /// Defines and instance of the execution policty.
  static constexpr auto execution_policy = execution_t{};

  /// Defines the traits of the solver.
  using solver_traits_t = 
    solver::SolverTraits<loader_t, reconstructor_t, flux_solver_t>;

  /// Defines the type of the solver based on the desired implementation.
  using solver_t = 
    std::conditional_t<SolverImpl == solver::Type::split
    , solver::SplitSolver<solver_traits_t, spacial_dims>
    , solver::UnsplitSolver<solver_traits_t, spacial_dims>
    >;
};

}} // namespace fluid::dim