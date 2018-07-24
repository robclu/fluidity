//==--- fluidity/tests/2d_shock_bubble_validation.cu ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  1d_shock_bubble_validation.cu
/// \brief This file defines a validation test for two dimensions where a shock
///        wave interacts with a bubble.
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

  constexpr auto res             = real_t{0.01};
  constexpr auto size_x          = real_t{1.6};
  constexpr auto size_y          = real_t{1.0};
  constexpr auto shock_start     = real_t{0.1};
  constexpr auto bubble_centre_x = real_t{0.4};
  constexpr auto bubble_centre_y = real_t{0.5}; 
  constexpr auto bubble_radius   = real_t{0.2};

  simulator->configure_resolution(res)
           ->configure_dimension(fluid::dim_x, 0.0, size_x)
           ->configure_dimension(fluid::dim_y, 0.0, size_y)
           ->configure_sim_time(0.4)
           ->configure_cfl(0.9)
           ->configure_max_iterations(20);

  // Returns the value based on whether the pos is inside the bubble,
  // or before or after the shock wave.
  auto shock_bubble_val = [&] (const auto& pos, auto in, auto pre, auto post)
  {
    const auto x = pos[0] * size_x - bubble_centre_x;
    const auto y = pos[1] * size_y - bubble_centre_y;

    const auto inside = std::sqrt(x*x + y*y) < bubble_radius;

    return inside ? in : pos[0] * size_x < shock_start ? pre : post;
  };

  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return shock_bubble_val(pos, 1.0, 3.81062, 1.0);
      } 
    },
    {
      "p", [&] (const auto& pos)
      {
        return shock_bubble_val(pos, 0.1, 9.98625, 1.0);
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return shock_bubble_val(pos, 0.0, 2.5745, 0.0);
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return 0.0;
      }
    }
  });

  simulator->simulate();
  simulator->write_results_separate_raw("2d_shock_bubble");
  simulator->write_results("2d_shock_bubble_results");
}