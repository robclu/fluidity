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
#include <fluidity/material/material_iterator.hpp>
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

/// Specialization of the SimulationData class when the execution policy is for
/// GPUs.
/// \tparam Traits The traits for the simulation.
template <typename Traits, typename Material>
class MultimaterialSimData<Traits, Material, exec::DeviceKind::gpu> {
 public:
  /// Defines the execution policy for the data.
  using exec_t              = exec::gpu_type;

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

  host_storage_t      _states;        //!< States input for an iteration.
  device_storage_t    _states_in;     //!< States output for an iteration.
  device_storage_t    _states_out;    //!< States output for an iteration.
  wavespeed_storage_t _wavespeeds;    //!< Wavespeeds for the simulation.
  material_t          _material;      //!< The material for the simulation.
  material_t          _material_out;  //!< The material used for outputting.
  solver_t            _solver;

 public:
  //==--- [New Interface] --------------------------------------------------==//

  /// Evolves the material data using the solver for the material, the timestep
  /// factor ($\frac{dt}{dh}$) for the simulation, and the boundary setter. This
  /// performs the evolution such that the updated data is the output data, thus
  /// a call to `swap_in_out_data()` needs to be performed after the iteration
  /// if further
  /// iterations are required.
  /// \param[in] dtdh            The timestep factor for evolution.
  /// \param[in] boundary_setter The boundary setter for the simulation.
  /// \tparam    T               The type of the timestep factor.
  /// \tparam    BoundarySetter  The type of the boundary setter.
  template <typename T, typename BoundarySetter>
  void evolve(T dtdh, const BoundarySetter& boundary_setter)
  {
    auto material_solver = solver_t();
    auto input_it        = input_iterator();
    material_solver.set_grid_sizes(input_it);
    material_solver.solve(std::move(input_it)         ,
                          std::move(output_iterator()),
                          _material.eos()             ,
                          dtdh                        ,
                          boundary_setter             );
  }

  /// Returns a MaterialIteratorWrapper which holds the equation of state for
  /// the material, and iterators to the levelset and state data for the
  /// material.
  auto get_iterator_wrapper() const
  {
    return make_material_iterator_wrapper(
      _material.eos()                      ,
	    _material.levelset().multi_iterator(),
	    _states_in.multi_iterator()          );
  }

  /// Returns a material iterator over the levelset and state data.
  auto material_iterator() const
  {
    return make_material_iterator(_material.eos()                      ,
                                  _material.levelset().multi_iterator(),
                                  _states_in.multi_iterator()          );
  }

  /// Initializes the data.
  void initialize()
  {
    _wavespeeds.resize(_states.total_size());
    _solver.set_grid_sizes(_states.multi_iterator());
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
    _material_out.levelset().resize_dim(dim, elements);
  }

  /// Sets the state data for the states which are inside the levelset.
  /// \param[in] components The components which define how each value in the
  ///                       state (i.e density, pressure, etc) are set.
  /// \tparam    Components The types of the components.
  template <typename... Components>
  void set_state_data(Components&&... components)
  {
    auto cs = make_tuple(std::forward<Components>(components)...);

    // The state is always primitive since it's easier
    // to set the data from a user's perspective.
    auto setter_state = typename traits_t::primitive_t();

    // Set the values of the components in the setter state.
    for_each(cs, [&] (auto& component)
    {
      setter_state.set_component(component);
    });

    // Use the levelset to set the values for the states which are
    // inside the levelset. This is done using the setter_state.
    state::set_states(
      _states_in.multi_iterator()          ,
      _material.levelset().multi_iterator(),
      setter_state                         ,
      _material.eos()                      );

    // Make sure that the filled data is available 
    // on both the host and the device.
    sync_device_to_host();
  }

  /// Swaps the input and output data for successive iterations.
  void swap_material_state_in_out_data()
  {
    auto in  = &(*input_iterator());
    auto out = &(*output_iterator());
    _states_in.reset_data(out);
    _states_out.reset_data(in);
  }

  /// Swaps the input and output levelset data for the material. This should be
  /// called after the levelsets have been updated if successive iterations are
  /// required so that the result of the update is the input for the next
  /// iteration.
  void swap_material_levelset_in_out_data()
  {
    auto in  = &(*_material.levelset().multi_iterator());
    auto out = &(*_material_out.levelset().multi_iterator());
    _material.levelset().reset_data(out);
    _material_out.levelset().reset_data(in);
  }

  /// Ensures that the host data is synchronized with the device data by copying
  /// the data which is on the device to the data on the host. If the host data
  /// more recent, this will overwrite that data.
  void sync_device_to_host()
  {
    _states = _states_in.as_host();
  }

  //==--- [Old Interface] --------------------------------------------------==//

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

  /// Returns a constant reference to the material.
  auto& material_out()
  {
    return _material_out;
  }

  /// Returns a constant reference to the material.
  const auto& material_out() const
  {
    return _material_out;
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

  template <typename MA, typename MB>
  void swap_material_levelsets(MA&& ma, MB&& mb)
  {
    _material.levelset().reset_data(&(*ma));
    _material_out.levelset().reset_data(&(*mb));
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
