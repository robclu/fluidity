//==--- fluidity/tests/2d_liska_case_15_validation.cu --------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  2d_liska_case_15_validation.cu
/// \brief This file defines a validation test for 2D solvers where the input
///        data is a 2D riemann problem.
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
  constexpr auto res     = real_t{0.0025};
  constexpr auto size_x  = real_t{1.0};
  constexpr auto size_y  = real_t{1.0};

  fluid::sim::sim_option_manager_t sim_manager;
  auto simulator = sim_manager.create_default();
  //sim_manager.configure(settings);

  simulator->configure_resolution(res);
  simulator->configure_dimension(fluid::dim_x, 0.0, size_x);
  simulator->configure_dimension(fluid::dim_y, 0.0, size_y);
  simulator->configure_sim_time(0.2);
  simulator->configure_cfl(0.9);

  simulator->fill_data({
    {
      "rho", [&] (const auto& pos)
      {
        return pos[1] > 0.5
        ? pos[0] < 0.5
          ? 0.5197 : 1.0        // (top left) | (top right)
        : pos[0] < 0.5          // -----------|-------------
          ? 0.8 : 0.5313;       // (bot left) | (bot right)
      }
    },
    {
      "p", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? 0.4 : 1.0           // (top left) | (top right)
          : pos[0] < 0.5          // -----------|-------------
            ? 0.4 : 0.4;          // (bot left) | (bot right)
      }
    },
    {
      "v_x", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? -0.6259 : 0.1        // (top left) | (top right)
          : pos[0] < 0.5           // -----------|-------------
            ? 0.1     : 0.1;       // (bot left) | (bot right)
      }
    },
    {
      "v_y", [&] (const auto& pos)
      {
        return pos[1] > 0.5
          ? pos[0] < 0.5
            ? -0.3 : -0.3           // (top left) | (top right)
          : pos[0] < 0.5            // -----------|-------------
            ? -0.3 : 0.4276;        // (bot left) | (bot right)
      }
    }
  });

  simulator->simulate();
  simulator->write_results_separate_raw("2d_liska_case_15");
}