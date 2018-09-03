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
/// \file  2d_shock_bubble_validation.cu
/// \brief This file defines a validation test for two dimensions where a shock
///        wave interacts with a bubble.
//
//==------------------------------------------------------------------------==//
/*
#include <fluidity/flux_method/flux_methods.hpp>
#include <fluidity/limiting/limiters.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/reconstructors.hpp>
#include <fluidity/simulator/generic_simulator.hpp>
#include <fluidity/state/state.hpp>
*/

#include <fluidity/setting/settings.hpp>
#include <fluidity/simulator/simulator_impl.hpp>
#include <iostream>
#include <memory>

using namespace fluid;

using real_t = double;

int main(int argc, char** argv)
{
  constexpr auto res             = real_t{0.004};
  constexpr auto size_x          = real_t{1.6};
  constexpr auto size_y          = real_t{1.0};
  constexpr auto shock_start     = real_t{0.18};
  constexpr auto bubble_centre_x = real_t{0.4};
  constexpr auto bubble_centre_y = real_t{0.5}; 
  constexpr auto bubble_radius   = real_t{0.2};

  if (argc < 2)
  {
    printf("Invalid usage, usage is:\n\n\t./<app> path-to-settings-file : %i\n",
    argc);
  }
  auto settings = fluid::setting::Settings(argv[1]);

  fluid::sim::sim_option_manager_t sim_manager;
  sim_manager.configure(settings);

  // This compiles a lot quicker, so use in debug mode:
  auto simulator = sim_manager.create_default();

  // This compiles all functionality, so use in release mode:
  //auto simulator = sim_manager.create();

  simulator->configure_resolution(res);
  simulator->configure_dimension(fluid::dim_x, 0.0, size_x);
  simulator->configure_dimension(fluid::dim_y, 0.0, size_y);
  simulator->configure_sim_time(0.15);
  simulator->configure_cfl(0.9);

  // Returns the value based on whether the pos is inside the bubble,
  // or before or after the shock wave.
  auto shock_bubble_val = [&] (const auto& pos, auto in, auto pre, auto post)
  {
    const auto x      = pos[0] * size_x - bubble_centre_x;
    const auto y      = pos[1] * size_y - bubble_centre_y;
    const auto inside = std::sqrt(x*x + y*y) < bubble_radius;

    return inside ? in : pos[0] * size_x < shock_start ? pre : post;
  };

  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return shock_bubble_val(pos, 0.1, 3.81062, 1.0);
        //return shock_bubble_val(pos, 1.0, 3.81062, 1.0);
      } 
    },
    {
      "p", [&] (const auto& pos)
      {
        return shock_bubble_val(pos, 1.0, 9.98625, 1.0);
        //return shock_bubble_val(pos, 0.1, 9.98625, 1.0);
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
}
