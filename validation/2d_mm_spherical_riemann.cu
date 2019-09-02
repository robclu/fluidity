//==--- fluidity/validation/2d_mm_spherical_riemann.cu ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  2d_mm_spherical_riemann.cu
/// \brief This file defines a validation test for a 2D spherical Riemann
///        problem, presented in the ghost fluid method of \cite Sambasivan2009.
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
  simulator->configure_dimension(fluid::dim_x, 0.0, 1.5);
  simulator->configure_dimension(fluid::dim_y, 0.0, 1.2);
  simulator->configure_sim_time(0.8);
  simulator->configure_cfl(0.9);
  simulator->configure_max_iterations(1000);

  if (argc >= 2) {
    auto arg_index = 1;
    while (arg_index + 1 <= argc) {
      switch (hash(argv[arg_index])) {
        case hash("res"):
          simulator->configure_resolution(atof(argv[++arg_index]));
          simulator->configure_dimension(fluid::dim_x, 0.0, 1.5);
          simulator->configure_dimension(fluid::dim_y, 0.0, 1.2);
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

  constexpr auto center_x = real_t{0.5};
  constexpr auto center_y = real_t{0.5};
  constexpr auto radius   = real_t{0.2};

  // Left side material, air:
  simulator->add_material(
    fluid::material::IdealGas<real_t>{1.4},
    [&] fluidity_host_device (auto it, auto& positions) {
      const auto x = positions[0] - center_x;
      const auto y = positions[1] - center_y;
      *it = std::sqrt(x * x + y * y) - radius;
    },
    1.0_rho, 5.0_p, 0.0_v_x 
  );
  // Right side material, helium:
  simulator->add_material(
    fluid::material::IdealGas<real_t>{1.4},
    [&] fluidity_host_device (auto it, auto& positions) {
      const auto x = positions[0] - center_x;
      const auto y = positions[1] - center_y;
      *it = -(std::sqrt(x * x + y * y) - radius);
    },
    1.0_rho, 1.0_p, 0.0_v_x 
  );


  simulator->simulate_mm();
  simulator->print_results();
  simulator->write_results("2d_mm_spherical_riemann");
}
