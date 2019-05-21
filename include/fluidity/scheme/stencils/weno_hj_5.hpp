//==--- fluidity/scheme/stencils/weno_hj_5.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  weno_hj_5.hpp
/// \brief This file defines an implementation of the HJ WENO 5 method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_STENCILS_HJ_WENO_5_HPP
#define FLUIDITY_SCHEME_STENCILS_HJ_WENO_5_HPP

#include "../interfaces/stencil.hpp"
#include <algorithm>

namespace fluid   {
namespace scheme  {
namespace stencil {

/// This provides an implementation of a WENO scheme which is 5th order for
/// Hamilton-Jacobi equations and 4th order for conservation laws. It
/// implements the stencil interface. Details of the implementation can be
/// found in:
///
///   [Gibou : A review of level-set methods and some recent applications]
///   (www.sciencedirect.com/science/article/pii/S0021999117307441).
struct HJWeno5 : public Stencil<HJWeno5> {
 private:
  /// Defines the maximum offset required by the stencil.
  static constexpr auto max_offset = 3;

 public:
  /// Returns the width of the stencil.
  fluidity_host_device constexpr auto width() const { return max_offset; }

  /// Implementation of the forward difference for the WENO scheme.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] h      The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto backward_deriv_impl(It&& it, T h, Dim dim) const
  {
    const auto d1 = it.offset(int{-2}, dim).backward_diff(dim) / h;
    const auto d2 = it.offset(int{-1}, dim).backward_diff(dim) / h;
    const auto d3 = it.backward_diff(dim) / h;
    const auto d4 = it.forward_diff(dim) / h;
    const auto d5 = it.offset(int{1}, dim).forward_diff(dim) / h;
    return diff(d1, d2, d3, d4, d5);
  }

  /// Implementation of the forward difference for the WENO scheme.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] h      The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim>
  fluidity_host_device auto forward_deriv_impl(It&& it, T h, Dim dim) const
  {
    const auto d1 = it.offset(int{2}, dim).forward_diff(dim) / h;
    const auto d2 = it.offset(int{1}, dim).forward_diff(dim) / h;
    const auto d3 = it.forward_diff(dim) / h;
    const auto d4 = it.backward_diff(dim) / h;
    const auto d5 = it.offset(int{-1}, dim).backward_diff(dim) / h;
    return diff(d1, d2, d3, d4, d5);
  }

 private:
  /// A generic implementation of the difference operator using the finite
  /// differences, returning the difference $D_{dim}^{\pm}\phi$ where the $\pm$
  /// depends on the finite differences provided.
  /// \param[in] d1 The first finite difference.
  /// \param[in] d2 The second finite difference.
  /// \param[in] d3 The third finite difference.
  /// \param[in] d4 The fourth finite difference.
  /// \param[in] d5 The fifth finite difference.
  template <typename T>
  fluidity_host_device auto diff(T&& d1, T&& d2, T&& d3, T&& d4, T&& d5) const
  {
    const auto phi_1 = 2.0 * d1 - 7.0 * d2 + 11.0 * d3;
    const auto phi_2 = -d2      + 5.0 * d3 + 2.0 * d4;
    const auto phi_3 = 2.0 * d3 + 5.0 * d4 - d5;

    // Compute the coefficients:
    auto alpha_1 = 13.0 / 12.0 * std::pow(d1 - 2.0 * d2 + d3, 2)
                 + std::pow(d1 - 4.0 * d2 + 3.0 * d3, 2) / 4.0;
    auto alpha_2 = 13.0 / 12.0 * std::pow(d2 - 2 * d3 + d4 , 2)
                 + std::pow(d2 - d4, 2) / 4.0;
    auto alpha_3 = 13.0 / 12.0 * std::pow(d3 - 2.0 * d4 + d5, 2)
                 + std::pow(3.0 * d3 - 4.0 * d4 + d5, 2);

    // Compute the error term:
    const auto e = 10.0e-6
                 * std::max(
                     std::max(
                       std::max(
                         std::max(d1*d1, d2*d2), d3 * d3), d4 * d4), d5 * d5)
                 + 10.0e-99;

    // Update the coefficients:
    alpha_1 = 0.1 / std::pow(alpha_1 + e, 2);
    alpha_2 = 0.6 / std::pow(alpha_2 + e, 2);
    alpha_3 = 0.3 / std::pow(alpha_3 + e, 2);

    // Finally, compute the difference. This incorperates the denominator from
    // the phis, which reduces the number of arithmetic operations.
    const auto w_denom = (alpha_1 + alpha_2 + alpha_3) * 6.0;
    return (alpha_1 * phi_1 + alpha_2 * phi_2 + alpha_3 * phi_3) / w_denom;
  }
};

}}} /// namespace fluid::scheme::stencil

#endif // FLUIDITY_SCHEME_STENCIL_HJ_WENO_5 _HPP
