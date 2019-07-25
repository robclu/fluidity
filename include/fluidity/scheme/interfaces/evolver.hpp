//==--- fluidity/scheme/interfaces/evolver.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  evolver.hpp
/// \brief This file defines an interface for a evolver -- an object which
///        evolves data using a 
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_EVOLVER_HPP
#define FLUIDITY_SCHEME_INTERFACES_EVOLVER_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

/// The Evolver class provides an interface to which numerical methods must
/// conform. The specific implementation of the Evolver is defined by template
/// type. The __only__ purpose of the evolver interface is to take input data
/// and to compute the updated value of that input data, and then to use it to
/// set output data to that value.
/// \tparam EvolverImpl The implementation of the evolver interface.
template <typename EvolverImpl>
class Evolver {
  /// Defines the type of the stencil implementation.
  using impl_t = EvolverImpl;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl() {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  constexpr auto width() const {
    return impl()->width();
  }

  /// Loads valid data into the padding cells for the \p shared_in shared memory
  /// iterator, from the \p global_it iterator over the global data, and the
  /// boundaries defined by \p bounds.
  ///
  /// \pre The the shared_it points to the data at the block thread indices and
  ///      the global_it points to the global data at the global thread indices.
  ///
  /// \param[in] global_it      The global memory iterator to load from.
  /// \param[in] shared_it      The shared memory iterator to load.
  /// \param[in] bounds         The information for the boundaries.
  /// \tparam    GlobalIterator The global memory iterator type.
  /// \tparam    SharedIterator The shared memory iterator type.
  /// \tparam    BoundContianer The type of the boundaries.
  template <
    typename GlobalIterator,
    typename SharedIterator,
    typename BoundContainer
  >
  fluidity_host_device auto load_padding(
    GlobalIterator&& global_it,
    SharedIterator&& shared_it,
    BoundContainer&& bounds 
  ) const -> void {
    impl()->load_padding_impl(
      std::forward<GlobalIterator>(global_it),
      std::forward<SharedIterator>(shared_it),
      std::forward<BoundContainer>(bounds)
    );
  }

  /// This function evolves the data by updating the \p out_it data using
  /// the \p in_it data. The method used to do so is defined by the
  /// implemenation of the interface.
  ///
  /// This interface allows different input and output iterator types since it
  /// is possible that this will be the case for multi-materials. So long as the
  /// \p out_it data can be set from the \p in_it data, this is okay.
  ///
  /// \param[in] in_it          The iterable input data to use to evolve.
  /// \param[in] out_it         The iteratable output data to update.
  /// \param[in] dt             The time resolution to use for the update.
  /// \param[in] dh             The spacial resolution to use for the update.
  /// \param[in] func_or_it     An additional functor/iterator for the update.
  /// \param[in] args           Additional arguments for the evolution.
  /// \tparam    InIterator     The type of the input iterator.
  /// \tparam    OutIterator    The type of the output iterator.
  /// \tparam    T              The type of the timestep and resolution.
  /// \tparam    FuncOrIt       The type of the functor/additional iterator.
  /// \tparam    Args           The types of any additional arguments.
  template <
    typename    InIterator   ,
    typename    OutIterator  ,
    typename    T            ,
    typename    FuncOrIt     ,
    typename... Args
  >
  fluidity_host_device auto evolve(
    InIterator&&    in_it     ,
    OutIterator&&   out_it    ,
    T               dt        ,
    T               dh        ,
    FuncOrIt&&      func_or_it,
    Args&&...       args 
  ) const -> void {
    static_assert(
      is_multidim_iter_v<InIterator>,
      "Input iterator must be a multidimensional iterator!"
    );
    static_assert(
      is_multidim_iter_v<OutIterator>, 
      "Output iterator must be a multidimensional iterator!"
    );

    return impl()->evolve_impl(
      std::forward<InIterator>(in_it)   ,
      std::forward<OutIterator>(out_it) ,
      dt                                ,
      dh                                ,
      std::forward<FuncOrIt>(func_or_it),
      std::forward<Args>(args)...
    );
  }
};

//==--- Traits -------------------------------------------------------------==//

/// Returns true if the type T conforms to the Evolver interface.
/// \tparam T The type to check for conformity to the Evolver inteface.
template <typename T>
static constexpr auto is_evolver_v = 
  std::is_base_of<Evolver<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_INTERFACES_EVOLVER_HPP
