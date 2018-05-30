//==--- fluidity/simulator/parameters.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  parameters.hpp
/// \brief This file defines a class which stores the parameters of a
///        simulation.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_PARAMETERS_HPP
#define FLUIDITY_SIMULATOR_PARAMETERS_HPP

#include <type_traits>

namespace fluid {
namespace sim   {

/// The Parameters class provides the functionality for storing and updating
/// parameters of a simulation.
/// \tparam T   The type of the data for the simulation parameters.
template <typename T>
struct Parameters {
  /// Defines the type of the parameter data.
  using value_t = std::decay_t<T>;
  /// Defines the type of iteration variables.
  using iter_t  = std::size_t;

  /// Defines the default number of maximum iterations for the simulation.
  static constexpr iter_t default_iters = std::numeric_limits<iter_t>::max();

  /// Resolution of the computational domain. These parameters use the same
  /// resolution for each of the dimensions.
  value_t resolution = 0.1;
  /// CFL number for the simulation.
  value_t cfl        = 0.9;
  /// The time for which a simulation has been running.
  value_t run_time   = 0.0;
  /// The time for which a simulation must run until.
  value_t sim_time   = 0.0;
  /// The number of iterations performed during a simulation.
  iter_t iters       = 0;
  /// Maximum number of iterations for the simulation.
  iter_t max_iters   = default_iters;

  /// Updates the parameter values based on the \p max_wavespeed in the
  /// simulation domain.
  /// \param[in] max_wavespeed The maximum wavespeed in the simulation domain.
  void update_time_delta(value_t max_wavespeed)
  {
    _dt = cfl * resolution / max_wavespeed;
  }

  /// Updates the simulation information, which is the run time and the
  /// number of iterations for the simulation.
  void update_simulation_info()
  {
    run_time += _dt;
    iters    += iter_t{1};
  }

  /// Returns true if the simulation must still run.
  bool continue_simulation() const
  {
    return run_time < sim_time && iters < max_iters;
  } 

  /// Returns the value of the time delta for each iteration.
  fluidity_host_device value_t dt() const
  {
    return _dt;
  }

  /// Returns the value of the timestep divided by the resolution, which is
  /// commonly referred to as ($\lambda$) in the literature.
  fluidity_host_device value_t dt_dh() const
  {
    return _dt / resolution;
  }

 private:
  value_t _dt = 0.0; //!< Time delta for the simulation.
};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_PARAMETERS_HPP