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


#include <fluid/limiting/van_leer_limiter.hpp>
#include <fluid/reconstruction/muscl_reconstructor.hpp>
#include <fluid/solver/hllc_solver.hpp>
#include <fluid/simulator/generic_simulator.hpp>
#include <fluid/state/state.hpp>
#include <gtest/gtest.h>
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
using reconstructor_t = fluid::recin::MHReconstructor<fluid::limit::VanLeer>;

// Defines the traits of the 1d simulator:
using simulator1d_props_t =
  SimulationTraits
  < primitive1d_t
  , material_t
  , reconstructor_t
  , HllcSolver
  >;

// Defines the traits of the 2d simulator:
using simulator2d_props_t =
  SimulationTraits
  < primitive2d_t
  , material_t
  , reconstructor_t
  , HllcSolver
  >;

TEST(generic_simulator_tests, can_create_and_output_1d_data)
{
  using simulator_t = fluid::GenericSimulator<simulator1d_props_t>;

  auto simulator = std::make_unique<simulator_t>();

  simulator->fill_data({
    {
      "density", [] (const auto& pos)
      {
        pos[0] < 0.1 ? 0.1 : 1.0;
      }
    },
    {
      "pressure", [] (const auto& pos)
      {
        pos[0] < 0.5 : 0.5 : 1.0;
      }
    }

  })
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
