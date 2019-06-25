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
#include "../interfaces/evaluatable.hpp"
#include <fluidity/utility/portability.hpp>

namespace fluid   {
namespace scheme  {
namespace updater {

/// The RungeKutta3 class is an implementation of the Updater interface which
/// takes input and output data and performs a RK3 temporal update using the
/// evaluator to perform the spatial update.
/// \tparam Evaluator The evaluator for the spatial evaluation.
template <typename Evaluator>
struct RungeKutta3 : public Updater<RungeKutta3<Evaluator>> {
 public:
  /// Defines the type of the spatial evaluator for the RK3 updater.
  using evaluator_t = std::decay_t<Evaluator>;

  /// Returns the width required by the updater. This is the number of cells
  /// on a single side which are required.
  fluidity_host_device constexpr auto width() const
  {
    return evaluator_t().width();
  }

  /// Implemenation of the function to update the \p it_out data using the
  /// \p it_in data and the evaluator.
  ///
  /// \pre The iterators, \p it_in, \p it_out, must be offset to the cells which
  ///      will be used (\p it_in) and set (\p it_out). If \p f_v_it is
  ///      an iterator then it too must be correctly offset.
  ///
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  template <typename ItIn, typename ItOut, typename T>
  fluidity_host_device void 
  update_impl(ItIn&& it_in, ItOut&& it_out, T dt, T dh) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    const auto evaluator = evaluator_t();
    const auto phi_n     = *it_in;

    // Compute first temp evolution to t^{n+1}. We have to sync here because the
    // scheme in the next evolution needs to access the data from other threads.
    *it_out = *it_in - dt * evaluator.evaluate(it_in, dh);
    fluidity_syncthreads(); 

    // Evolve again in time to t^{n+2} and then use the result in a weighted
    // average to compute phi^{n+ 1/2}. Again we need to sync for the next
    // evolution.
    *it_in = 0.75 * phi_n 
           + 0.25 * (*it_out - dt * evaluator.evaluate(it_out, dh));
    fluidity_syncthreads(); 

    // Compute evolution to t^{n + 3/2}:
    *it_out = *it_in + dt * evaluator.evaluate(it_in, dh);

    // Finally, set the output data:
    *it_out = (phi_n / 3.0) + (2.0 / 3.0 * (*it_out));  
  }

  /// Implemenation of the function to update the \p it_out data using the
  /// \p it_in data and the evaluator.
  ///
  /// \pre The iterators, \p it_in, \p it_out, must be offset to the cells which
  ///      will be used (\p it_in) and set (\p it_out). If \p f_v_it is
  ///      an iterator then it too must be correctly offset./
  ///
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \param[in] f_v_it   A functor / additional data for the update.
  /// \param[in] args     Additional arguments for the functor.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  /// \tparam    FVIt     The type if the additional data iterator.
  /// \tparam    Args     The types of any additional arguments.
  template <typename    ItIn ,
            typename    ItOut,
            typename    T    ,
            typename    FVIt ,
            typename... Args >
  fluidity_host_device void update_impl(ItIn&&    it_in ,
                                        ItOut&&   it_out,
                                        T         dt    ,
                                        T         dh    ,
                                        FVIt&&    f_v_it,
                                        Args&&... args  ) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    const auto evaluator = evaluator_t();
    const auto phi_n     = *it_in;

    // Compute first temp evolution to t^{n+1}. We have to sync here because the
    // scheme in the next evolution needs to access the data from other threads.
    *it_out = *it_in - dt * evaluator.evaluate(it_in, dh, f_v_it, args...);
    fluidity_syncthreads(); 

    // Evolve again in time to t^{n+2} and then use the result in a weighted
    // average to compute phi^{n+ 1/2}. Again we need to sync for the next
    // evolution.
    *it_in = 0.75 * phi_n 
           + 0.25 * (*it_out - 
                     dt * evaluator.evaluate(it_out, dh, f_v_it, args...));
    fluidity_syncthreads(); 

    // Compute evolution to t^{n + 3/2}:
    *it_out = *it_in + dt * evaluator.evaluate(it_in, dh, f_v_it, args...);

    // Finally, set the output data:
    *it_out = (phi_n / 3.0) + (2.0 / 3.0 * (*it_out));  
  }
};

}}} // namespace fluid::scheme::updater

#endif // FLUIDITY_SCHEME_UPDATERS_RUNGE_KUTTA_3_HPP
