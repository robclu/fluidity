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

#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {
namespace sim   {

/// The SimualtionData class stores data which must be updated during a
/// simulation, as well as utilities for accessing and modifying the relevant
/// data. The class is specialized for CPU and GPU implementations.
/// \tparam SimTraits  The traits of the simulation
/// \tparam DeviceType The type of device on which to execute the simulation.
template <typename SimTraits, exec::DeviceKind DeviceType>
class SimulationData;

/// Specialization of the SimulationData class when the execution policy is for
/// CPUs.
/// \tparam Traits The traits of the simulation.
template <typename Traits>
class SimulationData<Traits, exec::DeviceKind::cpu> {

};

/// Specialization of the SimulationData class when the execution policy is for
/// GPUs.
/// \tparam Traits The traits for the simulation.
template <typename Traits>
class SimulationData<Traits, exec::DeviceKind::gpu> {
 private:
  /// Defines the type of the traits class.
  using traits_t            = Traits;
  /// Defines the type of the state data to store, always conservative.
  using state_t             = typename traits_t::conservative_t;
  /// Defines the data type used in the state vector.
  using value_t             = typename state_t::value_t;
  /// Defines the type of the container used to store the host state data.
  using host_storage_t      = HostTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used to store the state state.
  using device_storage_t    = DeviceTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used for storing wavespeed data.
  using wavespeed_storage_t = DeviceTensor<value_t, 1>;

  host_storage_t      _states;      //!< States input for an iteration.
  device_storage_t    _states_in;   //!< States output for an iteration.
  device_storage_t    _states_out;  //!< States output for an iteration.
  wavespeed_storage_t _wavespeeds;  //!< Wavespeeds for the simulation.

 public:
  /// Returns a reference to the host state data.
  auto& states()
  {
    return _states;
  }

  /// Returns a constant reference to the host state data.
  const auto& states() const
  {
    return _states;
  }

  /// Initializes the data.
  void initialize()
  {
    _wavespeeds.resize(_states.total_size());
  }

  /// Ensures that the data is on the host.
  void finalise()
  {
    _states = _states_in.as_host();
  }

  /// Swaps the input and output state data pointers. This should be called at
  /// the end of each iteration of the simulation.
  void swap_states()
  {
    std::swap(_states_in, _states_out);
  }

  /// Resets the data using the \p in and \p out iterators.
  template <typename In, typename Out>
  void reset(In&& in, Out&& out)
  {
    _states_in.reset_data(&(*in));
    _states_out.reset_data(&(*out));
  }

  /// Returns an iterator to the simulation input states.
  auto input_iterator()
  {
    return _states_in.multi_iterator();
  }

  /// Returns an iterator to the simulation output states.
  auto output_iterator()
  {
    return _states_out.multi_iterator();
  }

  /// Returns an iterator to the simulation wavespeeds.
  auto wavespeed_iterator()
  {
    return _wavespeeds.multi_iterator();
  }

  /// Returns a reference to the wavespeed tensor.
  auto& wavespeeds()
  {
    return _wavespeeds;
  }

  /// Returns a constant reference to the wavespeed tensor.
  const auto& wavespeeds() const
  {
    return _wavespeeds;
  }

  /// Resizes the state data to contain \p elements elements in the \p dim.
  /// dimension.
  /// \param[in] dim      The dimension to resize.
  /// \param[in] elements The number of elements for the dimension.
  void resize_dim(std::size_t dim, std::size_t elements)
  {
    _states.resize_dim(dim, elements);
    _states_in.resize_dim(dim, elements);
    _states_out.resize_dim(dim, elements);
  }

  /// Synchronizes the data, making sure that the device data is the same as the
  /// host data.
  void synchronize()
  {
    _states_in  = _states.as_device();
    _states_out = _states.as_device();
  }
};

}}

#endif // FLUIDITY_SIMULATOR_SIMULATION_DATA_HPP
