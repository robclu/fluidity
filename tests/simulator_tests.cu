//==--- fluidity/tests/simulator_tests.cpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulator_tests.cpp
/// \brief This file defines tests for general simulator functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/limiting/van_leer_limiter.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/muscl_reconstructor.hpp>
#include <fluidity/solver/hllc_solver.hpp>
#include <fluidity/simulator/generic_simulator.hpp>
#include <fluidity/state/state.hpp>
//#include <gtest/gtest.h>
#include <memory>

// Defines the type of data to use.
using real_t           = double;
// Defines a 1 dimensional primitive state.
using primitive1d_t   = fluid::state::primitive_t<real_t, 1>;
// Defines a 2 dimensional conservative state with one additional component.
using primitive2d_t   = fluid::state::primitive_t<real_t, 2, 1>;
// Defines the material type to use for the tests.
using material_t      = fluid::material::IdealGas<real_t>;
// Defines the type of the limiter for the simulations.
using reconstructor_t = 
  fluid::recon::MHReconstructor<real_t, fluid::limit::VanLeer>;
/// Defines the execution policy of the solver, CPU / GPU
using execution_t     = fluid::exec::gpu_type;

// Defines the traits of the 1d simulator:
using simulator1d_props_t =
  fluid::sim::SimulationTraits
  < primitive1d_t
  , material_t
  , reconstructor_t
  , fluid::solver::HllcSolver
  , fluid::solver::Type::split
  , execution_t
  >;

// Defines the traits of the 2d simulator:
using simulator2d_props_t =
  fluid::sim::SimulationTraits
  < primitive2d_t
  , material_t
  , reconstructor_t
  , fluid::solver::HllcSolver
  , fluid::solver::Type::split
  , execution_t
  >;

int main(int argc, char** argv)
{
  using simulator_t = fluid::sim::GenericSimulator<simulator1d_props_t>;

  auto simulator = std::make_unique<simulator_t>();
  simulator->configure_dimension(fluid::dim_x, { 0.05, 1.0 })
           ->configure_sim_time(0.1);

  simulator->fill_data({
    {
      "density", [] (const auto& pos)
      {
        return pos[0] < 0.3 ? 0.1 : 1.0;
      }
    },
    {
      "pressure", [] (const auto& pos)
      {
        return pos[0] < 0.5 ? 0.5 : 1.0;
      }
    },
    {
      "v_x", [] (const auto& pos)
      {
        return pos[0] < 0.25 ? 1.0 : 0;
      }
    }
  });

  simulator->simulate();

  std::cout << "Finished running\n";

  simulator->print_results();
}