//==--- fluidity/validation/1d_mm_shock_tube_gas_water.cu -- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  1d_mm_shock_tube_gas_water.cu
/// \brief This file defines a validation test for a 1D shock tube with gas and
///        water, presented in the ghost fluid method of \cite Sambasivan2009.
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

  simulator->configure_resolution(0.005);
  simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
  simulator->configure_sim_time(0.2);
  simulator->configure_cfl(0.9);
  simulator->configure_max_iterations(200);

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

  constexpr auto membrane = real_t{0.5};
  // Left side material, ideal gas:
  simulator->add_material(
    fluid::material::IdealGas<real_t>{2.5},
    [&] fluidity_host_device (auto it, auto& positions) {
      *it = positions[0] - membrane;
    },
    0.5_rho, 20000.0_p, 100.0_v_x 
  );
  // Right side material, water:
  simulator->add_material(
    fluid::material::StiffFluid<real_t>{4.4},
    [&] fluidity_host_device (auto it, auto& positions) {
      *it = membrane - positions[0];
    },
    1.0_rho, 1.0_p, 0.0_v_x 
  );


  simulator->simulate_mm();
  simulator->print_results();
  simulator->write_results("1d_mm_shock_tube_gas_water");
}
