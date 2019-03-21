//==--- fluidity/simulator/multimaterial_simulation_data.hpp -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multimaterial_simulation_data.hpp
/// \brief This file defines classes which store simulation state data for CPU
///        and GPU execution policies for multimaterial simulation.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATION_DATA_HPP
#define FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATION_DATA_HPP

#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/container/tuple.hpp>
#include <fluidity/state/state_initialization.hpp>

namespace fluid {
namespace sim   {

/// The SimualtionData class stores data which must be updated during a
/// simulation, as well as utilities for accessing and modifying the relevant
/// data. The class is specialized for CPU and GPU implementations.
/// \tparam SimTraits  The traits of the simulation
/// \tparam DeviceType The type of device on which to execute the simulation.
template <typename SimTraits, typename Material, exec::DeviceKind DeviceType>
class MultimaterialSimData;

/// Specialization of the SimulationData class when the execution policy is for
/// CPUs.
/// \tparam Traits The traits of the simulation.
template <typename Traits, typename Material>
class MultimaterialSimData<Traits, Material, exec::DeviceKind::cpu> {

};

/// A wrapper class for the material which wraps the equation of state and the
/// iterators for the material.
/// \tparam Eos     The equation of state for the material.
/// \tparam LSIt    The type of the iterator over the levelset data.
/// \tparam StateIt The type of the iterator over the state data.
template <typename Eos, typename LSIt, typename StateIt>
struct MaterialIteratorWrapper {
  Eos     eos;      
  LSIt    ls_iterator;
  StateIt state_iterator;

  template <typename E, typename L, typename S>
  MaterialIteratorWrapper(E&& e, L&& ls_it, S&& state_it)
  : eos(std::forward<E>(e)),
    ls_iterator(std::forward<L>(ls_it)),
    state_iterator(std::forward<S>(state_it)) {}
};

template <typename Eos, typename LSIt, typename StateIt>
auto make_material_iterator_wrapper(Eos&& eos, LSIt&& ls_it, StateIt&& state_it)
{
  using wrapper_t = MaterialIteratorWrapper<Eos, LSIt, StateIt>;
  return wrappper_t(std::forward<Eos>(eos), 
                    std::forward<LSIt>(ls_it),
                    std::forward<StateIt>(state_it));
}

/// Specialization of the SimulationData class when the execution policy is for
/// GPUs.
/// \tparam Traits The traits for the simulation.
template <typename Traits, typename Material>
class MultimaterialSimData<Traits, Material, exec::DeviceKind::gpu> {
 private:
  /// Defines the type of the traits class.
  using traits_t            = Traits;
  /// Defines the type of the material for the simulation.
  using material_t          = Material;
  /// Defines the type of the state data to store, always conservative.
  using state_t             = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t             = typename state_t::value_t;
  /// Defines the type of the container used to store the host state data.
  using host_storage_t      = HostTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used to store the state state.
  using device_storage_t    = DeviceTensor<state_t, state_t::dimensions>;
  /// Defines the type of the container used for storing wavespeed data.
  using wavespeed_storage_t = DeviceTensor<value_t, 1>;

  /// Defines the type of the solver for this material.
  using solver_t = typename traits_t::template mm_solver_t<material_t>;

  host_storage_t      _states;      //!< States input for an iteration.
  device_storage_t    _states_in;   //!< States output for an iteration.
  device_storage_t    _states_out;  //!< States output for an iteration.
  wavespeed_storage_t _wavespeeds;  //!< Wavespeeds for the simulation.
  material_t          _material;    //!< The material for the simulation.

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

  /// Returns a constant reference to the material.
  auto& material()
  {
    return _material;
  }

  /// Returns a constant reference to the material.
  const auto& material() const
  {
    return _material;
  }

  /// Returns a MaterialIteratorWrapper which holds the equation of state for
  /// the material, and iterators to the levelset and state data for the
  /// material.
  auto get_iterator_wrapper() const
  {
    using eos_t      = decltype(_material.eos());
    using ls_it_t    = decltype(_material.levelset().multi_iterator());
    using state_it_t = decltype(_states_in.multi_iterator());
    using wrapper_t  = MaterialIteratorWrapper<eos_t, ls_it_t, state_it_t>;
    return wrappper_t(_material.eos()                      ,
                      _material.levelset().multi_iterator(),
                      _states_in.multi_iterator()          );
  }

  /// Returns the solver for the material.
  auto solver() const
  {
    return solver_t();
  }

  /// Initializes the data.
  void initialize()
  {
    _wavespeeds.resize(_states.total_size());
  }

  /// Sets the state data for the states which are inside the levelset.
  template <typename... Components>
  void set_state_data(Components&&... components)
  {
    auto cs    = make_tuple(std::forward<Components>(components)...);

    // The state is always primitive, since it's easier to set the data from a
    // user's perspective.
    auto state = typename traits_t::primitive_t();

    for_each(cs, [&] (auto& component)
    {
      state.set_component(component);
    });

    state::set_states(
      _states_in.multi_iterator()          ,
      _material.levelset().multi_iterator(),
      state                                ,
      _material.eos()                      );

    // Make sure that the filled data is available on both the host and the
    // device.
    sync_device_to_host();
  }

  /// Ensures that the host data is synchronized with the device data.
  void sync_device_to_host()
  {
    _states = _states_in.as_host();
  }

  /// Swaps the data using the \p a data for the input data, and the \p b data
  /// for the output data.
  /// \param[in] a The data to set the input state data to.
  /// \param[in] b The data to set the output state data to.
  /// \tparam    A The type of the a data.
  /// \tparam    B The type of the b data.
  template <typename A, typename B>
  void swap(A&& a, B&& b)
  {
    _states_in.reset_data(&(*a));
    _states_out.reset_data(&(*b));
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
    _material.levelset().resize_dim(dim, elements);
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

#endif // FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATION_DATA_HPP
