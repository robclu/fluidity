//==--- fluidity/tests/2d_liska_case_4_validation.cu ---------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  2d_liska_case_12_validation.cu
/// \brief This file defines a validation test for 2D solvers where the input
///        data is a 2D riemann problem.
//
//==------------------------------------------------------------------------==//

#include <fluidity/flux_method/flux_methods.hpp>
#include <fluidity/limiting/limiters.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/reconstructors.hpp>
#include <fluidity/simulator/generic_simulator.hpp>
#include <fluidity/state/state.hpp>
#include <iostream>
#include <memory>

using namespace fluid;

// Defines the type of data to use.
using real_t          = double;
// Defines a 1 dimensional primitive state.
using primitive2d_t   = state::primitive_t<real_t, 2>;
// Defines the material type to use for the tests.
using material_t      = material::IdealGas<real_t>;
/// Defines the type of the limiter to use.
using limiter_t       = limit::VanLeer<limit::cons_form_t>;
// Defines the type of the limiter for the simulations.
using reconstructor_t = recon::MHReconstructor<limiter_t>;
/// Defines the execution policy of the solver, CPU / GPU.
using execution_t     = fluid::exec::gpu_type;

// Defines the traits for the simulator to use the GPU.
using sim_traits_t =
  fluid::sim::SimulationTraits
  < primitive2d_t
  , material_t
  , reconstructor_t
  , flux::Hllc
  , solver::Type::split
  , execution_t
  >;

int main(int argc, char** argv)
{
  using simulator_t = fluid::sim::GenericSimulator<sim_traits_t>;
  auto simulator    = std::make_unique<simulator_t>();

  constexpr auto size  = real_t{1.0};
  constexpr auto cells = real_t{400};
  constexpr auto res   = size / cells;

  simulator->configure_resolution(res)
           ->configure_dimension(fluid::dim_x, 0.0, size)
           ->configure_dimension(fluid::dim_y, 0.0, size)
           ->configure_sim_time(0.25)
           ->configure_cfl(0.9);

  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return pos[1] > 0.5
        ? pos[0] < 0.5
          ? 1.0 : 0.5313        // (top left) | (top right)
        : pos[0] < 0.5          // -----------|-------------
          ? 0.8 : 1.0;          // (bot left) | (bot right)
      }
    },
    {
      "p", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? 1.0 : 0.4           // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 1.0 : 1.0;          // (bot left) | (bot right)
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? 0.7276 : 0.0        // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.0 : 0.0;          // (bot left) | (bot right)
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? 0.0 : 0.0           // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.0 : 0.7276;       // (bot left) | (bot right)
      }
    }
  });

  simulator->simulate();
  simulator->write_results_separate_raw("2d_liska_case_12");
}