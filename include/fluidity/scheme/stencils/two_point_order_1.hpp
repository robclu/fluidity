//==--- fluidity/scheme/stencils/two_point_order_1.hpp ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  two_point_order_1.hpp
/// \brief This file defines an implementation of the stencil interface using
///        a two point first order finite difference.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_STENCILS_TWO_POINT_ORDER_ONE_HPP
#define FLUIDITY_SCHEME_STENCILS_TWO_POINT_ORDER_ONE_HPP

#include "../interfaces/stencil.hpp"
#include <algorithm>

namespace fluid   {
namespace scheme  {
namespace stencil {

/// This provides and implementation of the stencil interface which computes
/// gradients in the forward and backward direction using a simple two point
/// stencil.
struct TwoPointOrder1 : public Stencil<TwoPointOrder1> {
 private:
  /// Defines the maximum offset required by the stencil.
  static constexpr auto max_offset = 1;

 public:
  /// Returns the width of the stencil.
  fluidity_host_device constexpr auto width() const { return max_offset; }

  /// Implementation of the forward difference for the first order stencil.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] dh     The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto backward_deriv_impl(It&& it, T dh, Dim dim) const
  {
    return it.backward_diff(dim) / dh;
  }

  /// Implementation of the forward difference for the WENO scheme.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] dh     The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto forward_deriv_impl(It&& it, T dh, Dim dim) const
  {
    return it.forward_diff(dim) / dh;
  }

  /// Implementation of the calculation of the quadratic coefficients iin the
  /// backward direction.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] dh     The delta for the stencil.
  /// \param[in] dim    The dimension to compute the coefficients for.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto quadratic_back_impl(It&& it, T dh, Dim dim) const
  {
    const auto alpha = T(1) / (dh*dh);
    const auto back  = *it.offset(-1, dim);
    const auto beta  = alpha * back;
    return Quadratic{alpha, T(-2) * beta, back * beta};
  }

  /// Implementation of the calculation of the quadratic coefficients iin the
  /// forward direction.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] dh     The delta for the stencil.
  /// \param[in] dim    The dimension to compute the coefficients for.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto quadratic_fwrd_impl(It&& it, T dh, Dim dim) const
  {
    const auto alpha = T(1) / (dh*dh);
    const auto fwrd  = *it.offset(1, dim);
    const auto beta  = alpha * fwrd;
    return Quadratic{alpha, T(-2) * beta, fwrd * beta};
  }
};

}}} // namespace fluid::scheme::stencil


#endif // FLUIDITY_SCHEME_STENCILS_TWO_POINT_ORDER_ONE_HPP
