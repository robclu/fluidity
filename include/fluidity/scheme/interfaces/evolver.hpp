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

  /// This function evolves the data by updating the \p it_out data using
  /// the \p it_in data. The method used to do so is defined by the
  /// implemenation of the interface.
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
  evolve(ItIn&& it_in, ItOut&& it_out, T dt, T dh, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    return impl()->evolve_impl(std::forward<ItIn>(it_in)  ,
                               std::forward<ItOut>(it_out),
                               dt                         ,
                               dh                         ,
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


#endif // FLUIDITY_SCHEME_INTERFACES_EVOLVER_HPP
