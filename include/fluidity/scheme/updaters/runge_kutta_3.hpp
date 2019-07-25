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
  fluidity_host_device constexpr auto width() const {
    return evaluator_t().width();
  }

  /// Implemenation of the function to update the \p out_it data using the
  /// \p in_it data and the evaluator.
  ///
  /// \pre The iterators, \p in_it, \p out_it, must be offset to the cells which
  ///      will be used (\p in_it) and set (\p out_it). Additionally, the
  ///      callind interface should ensure that they are multi-iterators with
  ///      `is_multidim_iter_v<>`.
  ///
  /// \param[in] in_it        The iterable input data to use to evolve.
  /// \param[in] out_it       The iteratable output data to update.
  /// \param[in] dt           The time resolution to use for the update.
  /// \param[in] dh           The spacial resolution to use for the update.
  /// \tparam    InIterator   The type of the input iterator.
  /// \tparam    OutIterator  The type of the output iterator.
  /// \tparam    T            The type of the timestep and resolution.
  template <typename InIterator, typename OutIterator, typename T>
  fluidity_host_device auto update_impl(
    InIterator&&  in_it ,
    OutIterator&& out_it,
    T             dt    ,
    T             dh
  ) const -> void {
    const auto evaluator = evaluator_t();
    const auto phi_n     = *in_it;

    // Compute first temp evolution to t^{n+1}. We have to sync here because the
    // scheme in the next evolution needs to access the data from other threads.
    *out_it = *in_it - dt * evaluator.evaluate(in_it, dh);
    fluidity_syncthreads(); 

    // Evolve again in time to t^{n+2} and then use the result in a weighted
    // average to compute phi^{n+ 1/2}. Again we need to sync for the next
    // evolution.
    *in_it = 0.75 * phi_n 
           + 0.25 * (*out_it - dt * evaluator.evaluate(out_it, dh));
    fluidity_syncthreads(); 

    // Compute evolution to t^{n + 3/2}:
    *out_it = *in_it - dt * evaluator.evaluate(in_it, dh);

    // Finally, set the output data:
    constexpr auto fact_13 = T(1.0) / T(3.0);
    constexpr auto fact_23 = T(2.0) * fact_13;
    *out_it = (fact_13 * phi_n) + (fact_23 * (*out_it));  
  }

  /// Implemenation of the function to update the \p out_it data using the
  /// \p in_it data, the evaluator, and the \p func_or_it functor or iterator to
  /// use in the evaluation.
  ///
  /// \pre The iterators, \p in_it, \p out_it, must be offset to the cells which
  ///      will be used (\p in_it) and set (\p out_it). If \p func_or_it is
  ///      an iterator then it too must be correctly offset. Additionally, the
  ///      callind interface should ensure that they are multi-iterators with
  ///      `is_multidim_iter_v<>`. 
  ///
  /// \param[in] in_it        The iterable input data to use to evolve.
  /// \param[in] out_it       The iteratable output data to update.
  /// \param[in] dt           The time resolution to use for the update.
  /// \param[in] dh           The spacial resolution to use for the update.
  /// \param[in] func_or_it   A functor / iterator over extra data for the update.
  /// \param[in] args         Additional arguments for the functor.
  /// \tparam    InIterator   The type of the input iterator.
  /// \tparam    OutIterator  The type of the output iterator.
  /// \tparam    T            The type of the timestep and resolution.
  /// \tparam    FuncOrIt     The type if the functor/extra iterator.
  /// \tparam    Args         The types of any additional arguments.
  template <
    typename    InIterator ,
    typename    OutIterator,
    typename    T          ,
    typename    FunctorOrIt,
    typename... Args 
  >
  fluidity_host_device auto update_impl(
    InIterator&&  in_it     ,
    OutIterator&& out_it    ,
    T             dt        ,
    T             dh        ,
    FunctorOrIt&& func_or_it,
    Args&&...     args 
  ) const -> void {
    const auto evaluator = evaluator_t();
    const auto phi_n     = *in_it;

    // Compute first temp evolution to t^{n+1}. We have to sync here because the
    // scheme in the next evolution needs to access the data from other threads.
    *out_it = *in_it - dt * evaluator.evaluate(in_it, dh, func_or_it, args...);
    fluidity_syncthreads(); 

    // Evolve again in time to t^{n+2} and then use the result in a weighted
    // average to compute phi^{n+ 1/2}. Again we need to sync for the next
    // evolution.
    *in_it = 0.75 * phi_n 
           + 0.25 * (*out_it - 
                     dt * evaluator.evaluate(out_it, dh, func_or_it, args...));
    fluidity_syncthreads(); 

    // Compute evolution to t^{n + 3/2}:
    *out_it = *in_it - dt * evaluator.evaluate(in_it, dh, func_or_it, args...);

    // Finally, set the output data:
    constexpr auto fact_13 = T(1.0) / T(3.0);
    constexpr auto fact_23 = T(2.0) * fact_13;
    *out_it = (fact_13 * phi_n) + (fact_23 * (*out_it));  
  }
};

}}} // namespace fluid::scheme::updater

#endif // FLUIDITY_SCHEME_UPDATERS_RUNGE_KUTTA_3_HPP
