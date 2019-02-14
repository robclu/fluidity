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
/// \file  2d_liska_case_4_validation.cu
/// \brief This file defines a validation test for 2D solvers where the input
///        data is a 2D riemann problem.
//
//==------------------------------------------------------------------------==//

#include <fluidity/setting/settings.hpp>
#include <fluidity/simulator/simulator_impl.hpp>
#include <iostream>
#include <memory>

using namespace fluid;

// Defines the type of data to use.
using real_t          = double;

int main(int argc, char** argv)
{
  constexpr auto res      = real_t{0.005};
  constexpr auto size_x   = real_t{1.0};
  constexpr auto size_y   = real_t{1.0};
  constexpr auto membrane = real_t{0.5};

  fluid::sim::sim_option_manager_t sim_manager;
  auto simulator = sim_manager.create_default();

  simulator->configure_resolution(res);
  simulator->configure_dimension(fluid::dim_x, 0.0, size_x);
  simulator->configure_dimension(fluid::dim_y, 0.0, size_y);
  simulator->configure_sim_time(0.25);
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
  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return pos[1] < membrane
          ? 1.0 : 0.125;
/*
          ? pos[0] < 0.5
            ? 1.1000 : 0.5065     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.5065 : 1.1000;    // (bot left) | (bot right)
*/
      }
    },
    {
      "p", [&] (const auto& pos)
      {
        return pos[1] < membrane
          ? 1.0 : 0.1;
/*
          ? pos[0] < 0.5
            ? 1.1000 : 0.3500     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.3500 : 1.1000;    // (bot left) | (bot right)
*/
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return pos[1] < membrane
          ? 0.75 : 0.0;
/*
          ? pos[0] < 0.5
            ? 0.8939 : 0.000      // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.8939 : 0.000;     // (bot left) | (bot right)
*/
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return 0.0;
/*
        return pos[1] < 0.5
          ? pos[0] < 0.5
            ? 0.8939 : 0.8939     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.000 : 0.000;      // (bot left) | (bot right)
*/
      }
    }
  });

/*
  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return pos[1] < 0.5
        ? pos[0] < 0.5
          ? 1.1000 : 0.5065     // (top left) | (top right)
        : pos[0] < 0.5          // -----------|-------------
          ? 0.5065 : 1.1000;    // (bot left) | (bot right)
      }
    },
    {
      "p", [&] (const auto& pos)
      {
        return pos[1] < 0.5
          ? pos[0] < 0.5
            ? 1.1000 : 0.3500     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.3500 : 1.1000;    // (bot left) | (bot right)
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return pos[1] < 0.5
          ? pos[0] < 0.5
            ? 0.8939 : 0.000      // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.8939 : 0.000;     // (bot left) | (bot right)
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return pos[1] < 0.5
          ? pos[0] < 0.5
            ? 0.8939 : 0.8939     // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.000 : 0.000;      // (bot left) | (bot right)
      }
    }
  });
*/
  simulator->simulate();
  simulator->write_results_separate_raw("2d_liska_case_4");
}