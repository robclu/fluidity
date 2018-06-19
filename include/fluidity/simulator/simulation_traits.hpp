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
#include <fluidity/solver/flux_solver.hpp>
#include <fluidity/solver/solver_traits.hpp>
#include <fluidity/solver/split_solver.hpp>
#include <fluidity/solver/unsplit_solver.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace sim   {

/// The SimulationTraits class defines the properties of a simulation.
///
/// \todo Change this so that multiple materials can be specified.
///
/// \tparam State           The state type to use for the simulation.
/// \tparam Material        The material for the simulation.
/// \tparam Reconstrucctor  The type of the reconstructor to use.
/// \tparam FluxMethod      The type of the flux method for dolving face fluxes.
/// \tparam Limiter         The type of the slope/flux limiter.
/// \tparam SolverImpl      Type of the solver (dimensionally split or unsplit).
/// \tparam ExecutionPolicy The target execution device type.
template < typename     State
         , typename     Material
         , typename     Reconstructor
         , typename     FluxMethod
         , solver::Type SolverImpl
         , typename     ExecutionPolicy = exec::default_type
         >
struct SimulationTraits {
 private:
  /// Defines the type of state used to define internal states.
  using state_internal_t = std::decay_t<State>;  
 public:
  /// Defines the type of material used for the simulation.
  using material_t      = std::decay_t<Material>;
  /// Defines the type of the reconstructor used for the simulation.
  using reconstructor_t = std::decay_t<Reconstructor>;
  /// Defines the type of the solver used for the simulation.
  using flux_method_t   = std::decay_t<FluxMethod>;
  /// Defines the type of the boundary loader.
  using loader_t        = solver::BoundaryLoader<reconstructor_t::width>;
  /// Defines the type of the execution policty for the simulation.
  using execution_t     = ExecutionPolicy;
  /// Defines the type of the data used by the state.
  using value_t         = typename state_internal_t::value_t;
  /// Defines the type of the face flux solver.
  using flux_solver_t   = 
    solver::FaceFlux<reconstructor_t, flux_method_t, material_t>;

  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto spacial_dims     = state_internal_t::dimensions;
  /// Defines and instance of the execution policty.
  static constexpr auto execution_policy = execution_t{};

  /// Defines the traits of the solver.
  //using solver_traits_t = 
  //  solver::SolverTraits<loader_t, reconstructor_t, flux_solver_t>;

  /// Defines the type for primitive states for this solver.
  using primitive_t = state::State< value_t
                                  , state::FormType::primitive
                                  , state_internal_t::dimensions
                                  , state_internal_t::additional_components
                                  , state_internal_t::storage_layout>;

  /// Defines a type for conservative states for the solver.
  using conservative_t = state::State< value_t
                                     , state::FormType::conservative
                                     , state_internal_t::dimensions
                                     , state_internal_t::additional_components
                                     , state_internal_t::storage_layout>;

  /// Defines the type of the state used by the solver. The solver always uses
  /// the conservative form of the state.
  using state_t = conservative_t;

  /// Defines the type of the solver based on the desired implementation.
  using solver_t = 
    std::conditional_t<SolverImpl == solver::Type::split
    , solver::SplitSolver<flux_solver_t, loader_t, spacial_dims>
    , solver::UnsplitSolver<flux_solver_t, loader_t, spacial_dims>
    >;
};

}} // namespace fluid::dim