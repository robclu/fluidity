//==--- fluidity/simulator/simulation_data.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_data.hpp
/// \brief This file defines classes which store simulation state data for CPU
///        and GPU execution policies.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATION_DATA_HPP
#define FLUIDITY_SIMULATOR_SIMULATION_DATA_HPP

namespace fluid {
namespace sim   {

/// The SimualtionData class stores data which must be updated during a
/// simulation, as well as utilities for accessing and modifying the relevant
/// data. The class is specialized for CPU and GPU implementations.
/// \tparam SimTraits  The traits of the simulation
/// \tparam DeviceType The type of device on which to execute the simulation.
template <typename SimTraits, exec::ExecutionKind DeviceType>
class SimulationData;

/// Specialization of the SimulationData class when the execution policy is for
/// CPUs.
template <typename SimTraits>
class SimulationData<SimTraits, exec::ExecutionKind::cpu> {

};

/// Specialization of the SimulationData class when the execution policy is for
/// GPUs.
template <typename SimTraits>
class SimulationData<exec::ExecutionKind::gpu> {
 private:
  /// Defines the type of the traits class.
  using traits_t  = SimTraits;
  /// Defines the type of the state data to store.
  using state_t   = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t   = typename state_t::value_t;
  /// Defines the type of the container used to store the host state data.
  using host_storage_t   = HostTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used to store the state state.
  using device_storage_t = DeviceTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used for storing wavespeed data.
  using wavespeed_t      = DeviceTensor<value_t, 1>;

  host_storage_t    _states;      //!< States input for an iteration.
  device_storage_t  _states_in;   //!< States output for an iteration.
  device_storage_t  _states_out;  //!< States output for an iteration.
  wavespeed_t       _wavespeeds;  //!< Wavespeeds for the simulation.

 public:
  /// Returns a reference to the host states which can be filled with data.
  host_storage_t& get_fillable_input_states()
  {
    return _states;
  }

  /// Returns an iterator to the simulation input states.
  auto input_iterator() const
  {
    return _states_in.multi_iterator();
  }

  /// Returns an iterator to the simulation output states.
  auto output_iterator() const
  {
    return _states_out.multi_iterator();
  }

  /// Returns an iterator to the simulation wavespeeds.
  auto wavespeed_iterator() const
  {
    return _wavespeeds.multi_iterator();
  }

  /// Resizes the state data to contain \p elements elements;
  void resize(std::size_t elements)
  {
    _states.resize(elements);
    _states_in.resize(elements);
    _states_out.resize(elements);
    _wavespeeds.resize(elements);
  }

  /// Synchronizes the data, making sure that the device data is the same as the
  /// host data.
  void synchronize()
  {
    _states_in  = _states;
    _states_out = _states;
  }
};

}}

#endif // FLUIDITY_SIMULATOR_SIMULATION_DATA_HPP
