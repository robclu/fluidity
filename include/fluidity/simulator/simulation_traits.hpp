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

namespace fluid {
namespace sim   {

/// The Simulation Traits class defines the properties of a simulation.
template < typename State
         , typename Material
         , typename Limiter
         , typename Reconstructor
         , typename Solver
         >
struct SimulationTraits {
  /// Defines the type of the state used for the simulation.
  using state_t         = std::decay_t<State>;
  /// Defines the type of material used for the simulation.
  using material_t      = std::decay_t<Material>;
  /// Defiens the type of the limiter used for the simulation.
  using limiter_t       = std::decay_t<Limiter>;
  /// Defines the type of the reconstructor used for the simulation.
  using reconstructor_t = std::decay_t<Reconstructor>;
  /// Defines the type of the solver used for the simulation.
  using solver_t        = std::decay_t<Sovler>;

  /// Defines the type of the data used by the state.
  using value_t         = typename state_t::value_t;

  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto spacial_dims = state_t::dimensions;
};

}} // namespace fluid::dim