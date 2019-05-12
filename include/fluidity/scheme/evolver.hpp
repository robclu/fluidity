//==--- fluidity/scheme/evolver.hpp ------------------------ -*- C++ -*- ---==//
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

#ifndef FLUIDITY_SCHEME_EVOLVER_HPP
#define FLUIDITY_SCHEME_EVOLVER_HPP

#include <fluidity/iterator/iterator_traits.hpp>

namespace fluid  {
namespace scheme {

/// The Evolver class provides an interface to which numerical methods must
/// conform. An Evolver uses a Scheme to compute the factor to use to update the
/// output data using the input data.
/// \tparam EvolverImpl The implementation of the evolver interface.
template <typename EvolverImpl>
class Evolver {
  /// Defines the type of the stencil implementation.
  using impl_t = EvolverImpl;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl()
  {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const
  {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  constexpr auto width() const
  {
    return impl()->width();
  }

  /// Overload of the function call operator to invoke the evolver on the data,
  /// updating the \p out data using the \p in data and the delta \p h.
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] h        The resolution to use for the update.
  /// \param[in] args     Additional arguments for the evolution.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename... Args>
  fluidity_host_device auto
  operator()(It&& it_in, It&& it_out, T h, Args&&...) const
  {
    static_assert(is_multidim_iter_v<It>, 
                  "Iterator must be a multidimensional iterator!");
    return impl()->invoke(std::forward<It>(it_in)    ,
                          std::forward<It>(it_out)   ,
                          h                          ,
                          std::forward<Args>(args)...);
  }
};

//==--- Traits -------------------------------------------------------------==//

/// Returns true if the type T conforms to the Evolver interface.
/// \tparam T The type to check for conformity to the Evolver inteface.
template <typename T>
static constexpr auto is_evolver_v = 
  std::is_base_of<Evolver<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_EVOLVER_HPP
