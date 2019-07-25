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
#include <fluidity/boundary/boundary_loading.hpp>

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
  fluidity_host_device constexpr auto width() const {
    return updater_t{}.width();
  }

  /// Loads valid data into the padding cells for the \p shared_in shared memory
  /// iterator, using the \p global_it iterator over the global data, and the
  /// boundaries defined by \p bounds.
  ///
  /// \pre The the shared_it points to the data at the block thread indices and
  ///      the global_it points to the global data at the global thread indices.
  ///
  /// \param[in] global_it      The global memory iterator to load from.
  /// \param[in] shared_it      The shared memory iterator to load.
  /// \param[in] bounds         The information for the boundaries.
  /// \tparam    GlobalIt       The global memory iterator type.
  /// \tparam    SharedIt       The shared memory iterator type.
  /// \tparam    BoundContianer The type of the boundaries.
  template <typename GlobalIt, typename SharedIt, typename BoundContainer>
  fluidity_host_device auto load_padding_impl(
    GlobalIt&&       global_it,
    SharedIt&&       shared_it,
    BoundContainer&& bounds 
  ) const -> void {
    constexpr auto w = updater_t().width();
    // There is nothing special to do for this evolver, we can simply use the
    // default loading to load the global data into the padding, or to set the
    // padding using the boundary information.
    boundary::load_padding_with_boundaries<w>(
      std::forward<GlobalIt>(global_it)   ,
      std::forward<SharedIt>(shared_it)   ,
      std::forward<BoundContainer>(bounds)
    );
  }

  /// Overload of the function call operator to invoke the evolver on the data,
  /// updating the \p out data using the \p in data and the time and space
  /// deltas.
  ///
  /// This interface allows different input and output iterator types since it
  /// is possible that this will be the case for multi-materials. So long as the
  /// \p it_out data can be set from the \p it_in data, this is okay.
  ///
  /// \param[in] it_in          The iterable input data to use to evolve.
  /// \param[in] it_out         The iteratable output data to update.
  /// \param[in] dt             The time resolution to use for the update.
  /// \param[in] dh             The spacial resolution to use for the update.
  /// \param[in] args           Additional arguments for the evolution.
  /// \param[in] func_or_it     The other iterator to use for the evolution.
  /// \tparam    IteratorIn     The type of the input iterator.
  /// \tparam    IteratorOut    The type of the output iterator.
  /// \tparam    T              The type of the timestep and resolution.
  /// \tparam    FuncOrIt       The type of the functor/other iterator.
  /// \tparam    Args           The types of any additional arguments.    
  template <
    typename    IteratorIn   ,
    typename    IteratorOut  , 
    typename    T            ,
    typename    FuncOrIt     ,
    typename... Args
  >
  fluidity_host_device auto evolve_impl(
    IteratorIn&&    it_in     ,
    IteratorOut&&   it_out    ,
    T               dt        ,
    T               dh        ,
    FuncOrIt&&      func_or_it,
    Args&&...       args
  ) const -> void {
    updater_t().update(
      std::forward<IteratorIn>(it_in)   ,
      std::forward<IteratorOut>(it_out) ,
      dt                                ,
      dh                                ,
      std::forward<FuncOrIt>(func_or_it),
      std::forward<Args>(args)...
    );
  }
};

}}} // namespace fluid::scheme::evolver

#endif // FLUIDITY_SCHEME_EVOLVERS_BASIC_EVOLVER_HPP
