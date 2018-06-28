//==--- fluidity/tests/1d_toro_case_2_validation.cu -------- -*- C++ -*- ---==//
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

#include <fluidity/flux_method/flux_methods.hpp>
#include <fluidity/limiting/limiters.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/basic_reconstructor.hpp>
#include <fluidity/reconstruction/muscl_reconstructor.hpp>
#include <fluidity/simulator/generic_simulator.hpp>
#include <fluidity/state/state.hpp>
#include <memory>

using namespace fluid;

// Defines the type of data to use.
using real_t          = double;
// Defines a 1 dimensional primitive state.
using primitive1d_t   = state::primitive_t<real_t, 1>;
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
  < primitive1d_t
  , material_t
  , reconstructor_t
  , flux::Force
  , solver::Type::split
  , execution_t
  >;

int main(int argc, char** argv)
{
  using simulator_t = fluid::sim::GenericSimulator<sim_traits_t>;

  auto simulator = std::make_unique<simulator_t>();
  simulator->configure_dimension(fluid::dim_x, { 0.01, 1.0 })
           ->configure_sim_time(0.15)
           ->configure_cfl(0.9);

  constexpr auto membrane = real_t{0.5};
  simulator->fill_data({
    {
      "rho", [] (const auto& pos)
      {
        return 1.0;
      }
    },
    { 
      "p", [] (const auto& pos)
      {
        return 0.4;
      }
    },
    {
      "v_x", [] (const auto& pos)
      {
        return pos[0] < membrane ? -2.0 : 2.0;
      }
    }
  });

  simulator->simulate();
  simulator->write_results("1d_toro_case_2_results");
}