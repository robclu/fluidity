//==--- fluidity/tests/1d_toro_case_5_validation.cu -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  1d_toro_case_2_validation.cu
/// \brief This file defines a validation test against the 1d toro test case 2.
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

  simulator->configure_resolution(0.005);
  simulator->configure_dimension(fluid::dim_x, 0.0, 1.0);
  simulator->configure_sim_time(0.035);
  simulator->configure_cfl(0.9);

  constexpr auto membrane = real_t{0.5};
  simulator->fill_data({
    {
      "rho", [] (const auto& pos)
      {
        return pos[0] < membrane ? 5.99924 : 5.99242;
      }
    },
    {
      "p", [] (const auto& pos)
      {
        return pos[0] < membrane ? 460.894 : 46.095;
      }
    },
    {
      "v_x", [] (const auto& pos)
      {
        return pos[0] < membrane ? 19.5975 : -6.19633;
      }
    }
  });

  simulator->simulate();
  simulator->write_results("1d_toro_case_5_results");
}