//==--- fluidity/scheme/updaters/runge_kutta_3.hpp --------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  runge_kutta_3.hpp
/// \brief This file defines an implementation of the RK3 method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_UPDATERS_RUNGE_KUTTA_3_HPP
#define FLUIDITY_SCHEME_UPDATERS_RUNGE_KUTTA_3_HPP

#include "../interfaces/updater.hpp"
#include "../interfaces/scheme.hpp"
#include <fluidity/utility/portability.hpp>

namespace fluid   {
namespace scheme  {
namespace updater {

template <typename Scheme>
struct RungeKutta3 : public Updater<RungeKutta3<Scheme>> {
 public:
  /// Defines the type of the scheme for the RK3 updater.
  using scheme_t = std::decay_t<Scheme>;

  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  fluidity_host_device constexpr auto width() const
  {
    return scheme_t().width();
  }

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
  update_impl(ItIn&& it_in, ItOut&& it_out, T dt, T dh, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    const auto scheme = scheme_t();
    const auto phi_n  = *it_in;

    // Compute first temp evolution to t^{n+1}. We have to sync here because the
    // scheme in the next evolution needs to access the data from other threads.
    *it_out = *it_in - dt * scheme(it_in, dh, args...);
    fluidity_syncthreads(); 

    // Evolve again in time to t^{n+2} and then use the result in a weighted
    // average to compute phi^{n+ 1/2}. Again we need to sync for the next
    // evolution.
    *it_in = 0.75 * phi_n 
           + 0.25 * (*it_out - dt * scheme(it_out, dh, args...));
    fluidity_syncthreads(); 

    // Compute evolution to t^{n + 3/2}:
    *it_out = *it_in + dt * scheme(it_in, dh, args...);

    // Finally, set the output data:
    *it_out = (phi_n / 3.0) + (2.0 / 3.0 * (*it_out));  
  }
};

}}} // namespace fluid::scheme::updater

#endif // FLUIDITY_SCHEME_UPDATERS_RUNGE_KUTTA_3_HPP
