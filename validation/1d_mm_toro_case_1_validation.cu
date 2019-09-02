//==--- fluidity/validation/1d_mm_toro_case_1_validation.cu  -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  1d_mm_toro_case_1_validation.cu
/// \brief This file defines a validation test against the 1d toro test case 1
//         but using the multi-material simulator.
//
//==------------------------------------------------------------------------==//

#include <fluidity/algorithm/hash.hpp>
#include <fluidity/setting/settings.hpp>
#include <fluidity/state/state_components.hpp>
#include <fluidity/simulator/multimaterial_simulator.hpp>
#include <iostream>
#include <memory>

using namespace fluid;

// Defines the type of data to use.
using real_t = double;

int main(int argc, char** argv) {
  using namespace fluid::state::components;

  fluid::sim::mm_sim_option_manager_t sim_manager;
  auto simulator = sim_manager.create_simulator();

  simulator->configure_resolution(0.0025);
  simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
  simulator->configure_sim_time(0.2);
  simulator->configure_cfl(0.9);

  if (argc >= 2) {
    auto arg_index = 1;
    while (arg_index + 1 <= argc) {
      switch (hash(argv[arg_index])) {
        case hash("res"):
          simulator->configure_resolution(atof(argv[++arg_index]));
          simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
          break;
        case hash("sim_time"):
          simulator->configure_sim_time(atof(argv[++arg_index]));
          break;
        case hash("iters"):
          simulator->configure_max_iterations(atoi(argv[++arg_index]));
          break;
        default:
          arg_index++;
      }
      arg_index++;
    }
  }

  constexpr auto membrane = real_t{0.3};
  // Left side material:
  simulator->add_material(
    fluid::material::IdealGas<real_t>{1.4},
    [&] fluidity_host_device (auto it, auto& positions) {
      *it = positions[0] - membrane;
    },
    1.0_rho, 1.0_p, 0.75_v_x 
  );

  // Right side material:
  simulator->add_material(
    fluid::material::IdealGas<real_t>{1.4},
    [&] fluidity_host_device (auto it, auto& positions) {
      *it = membrane - positions[0];
    },
    0.125_rho, 0.1_p, 0.0_v_x 
  );

  simulator->simulate_mm();
  simulator->print_results();
  simulator->write_results("1d_mm_toro_case_1_results");
}
