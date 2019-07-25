//==--- fluidity/scheme/interfaces/stencil.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  stencil.hpp
/// \brief This file defines an interface for a stencil for solving.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_STENCIL_HPP
#define FLUIDITY_SCHEME_INTERFACES_STENCIL_HPP

#include <fluidity/math/quadratic.hpp>
#include <type_traits>

namespace fluid  {
namespace scheme {

/// The Stencil class defines the interface to which a computational stencil
/// must conform. The implementation is provided by the template type. It simply
/// provides forwards and backward derivative interfaces for the stencil, from
/// which entire schemes can be built.
/// \tparam StencilImpl The type of the stencil implementation.
template <typename StencilImpl>
class Stencil {
  /// Defines the type of the stencil implementation.
  using impl_t = StencilImpl;

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
  /// Returns the width of the stencil.
  constexpr auto width() const
  {
    return impl()->width();
  }

  /// Interface for the backward derivative for the stencil.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] h      The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto backward_deriv(It&& it, T h, Dim dim) const
  {
    static_assert(is_multidim_iter_v<It>, 
                  "Iterator must be a multidimensional iterator!");
    return impl()->backward_deriv_impl(std::forward<It>(it), h, dim);
  }

  /// Interface for the forward derivative for the stencil.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] h      The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto forward_deriv(It&& it, T h, Dim dim) const
  {
    static_assert(is_multidim_iter_v<It>, 
                  "Iterator must be a multidimensional iterator!");
    return impl()->forward_deriv_impl(std::forward<It>(it), h, dim);
  }
};

//==--- Traits -------------------------------------------------------------==//

/// Returns true if the type T conforms to the Stencil interface.
/// \tparam T The type to check for conformity to the Stencil interface.
template <typename T>
static constexpr auto is_stencil_v = 
  std::is_base_of<Stencil<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_INTERFACES_STENCIL_HPP
