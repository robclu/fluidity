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

  /// Defines the default number of maximum iterations for the simulation.
  static constexpr value_t default_iters = std::numeric_limits<value_t>::max();

  /// Resolution of the computational domain. These parameters use the same
  /// resolution for each of the dimensions.
  value_t resolution = 0;
  /// CFL number for the simulation.
  value_t cfl        = 0;
  /// Total run time for the simulation.
  value_t run_time   = 0;
  /// Maximum number of iterations for the simulation.
  value_t max_iters  = default_iters;

  /// Updates the parameter values based on the \p max_wavespeed in the
  /// simulation domain.
  /// \param[in] max_wavespeed The maximum wavespeed in the simulation domain.
  fluidity_host_device update(value_t max_wavespeed)
  {
    dt = cfl * resolution / max_wavespeed;
  }

  /// Returns the value of the time delta for each iteration.
  fluidity_host_device value_t dt() const
  {
    return dt;
  }

  /// Returns the value of the timestep divided by the resolution, which is
  /// commonly referred to as ($\lambda$) in the literature.
  fluidity_host_device value_t dt_dh() const
  {
    return dt / resolution;
  }

 private:
  value_t dt = 0; //!< Time delta for the simulation.
};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_PARAMETERS_HPP