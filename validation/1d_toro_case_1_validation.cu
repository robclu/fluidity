//==--- fluidity/tests/1d_toro_case_1_validation.cu -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  1d_toro_case_1_validation.cu
/// \brief This file defines a validation test against the 1d toro test case 1.
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
  fluid::sim::sim_option_manager_t sim_manager;
  auto simulator = sim_manager.create_default();

  //simulator->configure_resolution(0.005);
  //simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
  //simulator->configure_sim_time(0.2);
  //simulator->configure_cfl(0.9);

  simulator->configure_resolution(0.0025);
  simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
  simulator->configure_sim_time(0.2);
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

  constexpr auto membrane = real_t{0.3};
  simulator->fill_data({
    {
      "rho", [] (const auto& pos)
      {
        return pos[0] < membrane ? 1.0 : 0.125;
      }
    },
    {
      "p", [] (const auto& pos)
      {
        return pos[0] < membrane ? 1.0 : 0.1;
      }
    },
    {
      "v_x", [] (const auto& pos)
      {
        return pos[0] < membrane ? 0.75 : 0.0;
      }
    }
  });

  simulator->simulate();
  simulator->print_results();
  simulator->write_results("1d_toro_case_1_results");
}
