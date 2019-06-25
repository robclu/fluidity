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

#include <fluidity/setting/settings.hpp>
#include <fluidity/simulator/simulator_impl.hpp>
#include <iostream>
#include <memory>

using namespace fluid;

using real_t = double;

int main(int argc, char** argv)
{
  // TODO: Remove this once the configuration from file is complete ...
  constexpr auto res             = real_t{0.005};
  //constexpr auto size_x          = real_t{1.6};
  constexpr auto size_x          = real_t{1.0};
  constexpr auto size_y          = real_t{1.0};
  constexpr auto shock_start     = real_t{0.2};
  constexpr auto bubble_centre_x = real_t{0.4};
  constexpr auto bubble_centre_y = real_t{0.5}; 
  constexpr auto bubble_radius   = real_t{0.2};
  constexpr auto membrane_y      = real_t{0.5};

  // Time to run the simulation until:
  auto sim_time = real_t{0.4};

  fluid::sim::sim_option_manager_t sim_manager;
/*
  TODO : Use this once completed implementation of simulation configuration
         from a file is complete ..
  if (argc < 2)
  {
    printf("Invalid usage, usage is:\n\n\t./<app> path-to-settings-file : %i\n",
    argc);
  }
  sim_manager.configure(fluid::setting::Settings::from_file(argv[1]));
*/


// Creating a simulator with all the possible options bloats compile-time
// significantly, so in debug mode we build the default, but in release mode the
// version with all functionality is built.
//#if !defined(NDEBUG)
auto simulator = sim_manager.create_default();

//#else
//  auto simulator = sim_manager.create();
//#endif
/*
  TOD: Remove all the configurations once the configuration from file is
        complete ...
*/
  simulator->configure_resolution(res);
  simulator->configure_dimension(fluid::dim_x, 0.0, size_x);
  simulator->configure_dimension(fluid::dim_y, 0.0, size_y);
  simulator->configure_sim_time(sim_time);
  simulator->configure_cfl(0.9);

  // Command line arguments
  if (argc >= 2)
  {
    simulator->configure_sim_time(atof(argv[1]));
  }
  if (argc >= 3)
  {
    simulator->configure_max_iterations(atoi(argv[2]));
  }

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
        return pos[1] < membrane_y
          ? pos[0] < 0.5
            ? 1.1000 : 0.5065     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.5065 : 1.1000;    // (bot left) | (bot right)
      }
    },
    {
      "p", [&] (const auto& pos)
      {
        return pos[1] < membrane_y
          ? pos[0] < 0.5
            ? 1.1000 : 0.3500     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.3500 : 1.1000;    // (bot left) | (bot right)
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return pos[1] < membrane_y
          ? pos[0] < 0.5
            ? 0.8939 : 0.000      // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.8939 : 0.000;     // (bot left) | (bot right)
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return pos[1] < membrane_y
          ? 0.8939 : 0.0000;
//            ? 0.8939 : 0.8939     // (top left) | (top right)
//          : pos[0] < 0.5          // -----------|-------------
//            ? 0.000 : 0.000;      // (bot left) | (bot right)
      }
    }
  });
/*
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
*/
  simulator->simulate();
  simulator->write_results_separate_raw("2d_shock_bubble");
}
