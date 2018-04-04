//==--- fluidity/simulator/simulation_updater.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.hpp
/// \brief This file defines a class which updates a simulation, and the
///        implementation is specialized for CPU and GPU execution policies.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP
#define FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP

#include "simulation_updater.cuh"

namespace fluid {
namespace sim   {

template <typename State, typename ExecutionPolicy>
struct SimulationUpdater {
  template <typename Storage,
            typename Loader ,
            typename Material>
  static void update(Storage&&  initial_states,
                     Storage&&  updated_states,
                     Loader&&   data_loader   ,
                     Material&& material      )
  {

  }
 private:
};

template <typename State>
struct SimulationUpdater<exec::gpu_type> {
  using state_t   = std::decay_t<State>;
  using storage_t = DeviceTensor<state_t, state_t::dimensions>;

  template <typename HostStorage>
  SimulationUpdater(const HostStorage& i_states, const HostStorage& u_states)
  : _initial_states(i_states), _updated_states(u_states) {}

  template <ttypename Loader, typename Material>
  void update(Loader&&   data_loader,
              Material&& material   )
  {

  }

 private:
  storage_t _initial_states;  //!< Initial states for simulation.
  storage_t _updated_states;  //< Updated states for simulation.
};


}} // namespace fluid::sim


#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP