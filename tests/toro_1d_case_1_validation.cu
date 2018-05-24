//==--- fluidity/tests/toro_1d_case_1_validation.cu -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  toro_1d_case_1_validation.cu
/// \brief This file defines a validation test against the 1d toro test case 1.
//
//==------------------------------------------------------------------------==//

#include <fluidity/limiting/van_leer_limiter.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/muscl_reconstructor.hpp>
#include <fluidity/solver/hllc_solver.hpp>
#include <fluidity/simulator/generic_simulator.hpp>
#include <fluidity/state/state.hpp>
#include <memory>

using namespace fluid;

// Defines the type of data to use.
using real_t           = double;
// Defines a 1 dimensional primitive state.
using primitive1d_t   = state::primitive_t<real_t, 1>;
// Defines the material type to use for the tests.
using material_t      = material::IdealGas<real_t>;
// Defines the type of the limiter for the simulations.
using reconstructor_t = recon::MHReconstructor<real_t, limit::VanLeer>;
/// Defines the execution policy of the solver, CPU / GPU.
using execution_t     = fluid::exec::gpu_type;

// Defines the traits for the simulator to use the GPU.
using sim_traits_gpu_t =
  fluid::sim::SimulationTraits
  < primitive1d_t
  , material_t
  , reconstructor_t
  , solver::HllcSolver
  , solver::Type::split
  , exec::gpu_type
  >;

// Defines the traits for the simulator to use the CPU.
using sim_traits_cpu_t =
  fluid::sim::SimulationTraits
  < primitive1d_t
  , material_t
  , reconstructor_t
  , solver::HllcSolver
  , solver::Type::split
  , exec::cpu_type
  >;

int main(int argc, char** argv)
{
  using simulator_t = fluid::sim::GenericSimulator<sim_traits_gpu_t>;

  auto simulator = std::make_unique<simulator_t>();
  simulator->configure_dimension(fluid::dim_x, { 0.01, 1.0 })
           ->configure_sim_time(0.1)
           ->configure_cfl(0.9);

  simulator->fill_data({
    {
      "density", [] (const auto& pos)
      {
        return pos[0] < 0.5 ? 1.0 : 0.125;
      }
    },
    {
      "pressure", [] (const auto& pos)
      {
        return pos[0] < 0.5 ? 1.0 : 0.1;
      }
    },
    {
      "v_x", [] (const auto& pos)
      {
        return pos[0] < 0.5 ? 0.75 : 0.0;
      }
    }
  });

  std::cout << "Initial data:\n";
  simulator->print_results();

  simulator->simulate();

  std::cout << "Final data:\n";
  simulator->print_results();
  simulator->write_results("toro_1d_case_1_results.txt");
}