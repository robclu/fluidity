//==--- fluidity/scheme/basic_evolver.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_evolver.hpp
/// \brief This file defines a simple implemenatation of the Evolver interface
///        which uses an updater to compute the update value for the data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_EVOLVERS_BASIC_EVOLVER_HPP
#define FLUIDITY_SCHEME_EVOLVERS_BASIC_EVOLVER_HPP

#include "../interfaces/evolver.hpp"

namespace fluid   {
namespace scheme  {
namespace evolver {

/// The BasicEvolver struct uses the Updater to compute the update for the data.
/// \tparam Updater The function which computes the update value for the data.
template <typename Updater>
struct BasicEvolver : public Evolver<BasicEvolver<Updater>> {
 public:
  /// Defines the type of the updater for the evolver.
  using updater_t = std::decay_t<Updater>;

  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  fluidity_host_device constexpr auto width() const
  {
    return updater_t{}.width();
  }

  /// Overload of the function call operator to invoke the evolver on the data,
  /// updating the \p out data using the \p in data and the time and space
  /// deltas.
  ///
  /// This interface allows different input and output iterator types since it
  /// is possible that this will be the case for multi-materials. So long as the
  /// \p it_out data can be set from the \p it_in data, this is okay.
  ///
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \param[in] args     Additional arguments for the evolution.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  /// \tparam    Args     The types of any additional arguments.    
  template <typename ItIn, typename ItOut, typename T, typename... Args>
  fluidity_host_device void
  evolve_impl(ItIn&& it_in, ItOut&& it_out, T dt, T dh, Args&&... args) const
  {
    const auto updater = updater_t{};
    updater.update(std::forward<ItIn>(it_in)  ,
                   std::forward<ItOut>(it_out),
                   dt                         ,
                   dh                         ,
                   std::forward<Args>(args)...);
  }
};

}}} // namespace fluid::scheme::evolver

#endif // FLUIDITY_SCHEME_EVOLVERS_BASIC_EVOLVER_HPP
