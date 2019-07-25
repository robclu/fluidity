//==--- fluidity/scheme/schemes/godunov_upwind.hpp --------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  godunov_upwind.hpp
/// \brief This file defines the implementation of a first order
///        Godunov upwind scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_HPP
#define FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_HPP

#include "../interfaces/solvable.hpp"
#include <fluidity/math/quadratic.hpp>

namespace fluid  {
namespace scheme {

/// The GodunovUpwind struct computes a first order spatial discretization of
/// the following Hamiltonian: $|\nabla \phi(x)|^2 = S(x)$, where $S(x)$ is the
/// source term. This struct does not implement the source component, but rather
/// implements the appropriate interfaces which allow different source
/// components to be used.
///
/// When evaluating the above Hamiltonian, the following is solved:
///
/// If $a <= 0$ then:
/// 
/// \begin{equation}
///   H(a, b, c, d) = 
///     \sqrt{
///       max(|max(a, 0)|^2, |min(b, 0)|^2) +
///       max(|max(c, 0)|^2, |min(d, 0)|^2)}
/// \end{equation}
///
/// else if $a > 0$ then:
///
/// \begin{equation}
///   H(a, b, c, d) = 
///     \sqrt{
///       max(|min(a, 0)|^2, |max(b, 0)|^2) +
///       max(|min(c, 0)|^2, |max(d, 0)|^2)}
/// \end{equation}
///
/// where in 2D, the values $a, b, c, d$ correspond to the derivatives 
/// $D_{ijk}^{x+}, D^{x-}, D^{y+}, D^{y-}$ which are be computed using the
/// stencil.
///
/// It also implements the Solvable interface, and when solving the
/// discretization, the value of $\phi(x)$ is returned which solves:
///
/// \begin{equation}
///   \sum_i^D [\max(D_{ijk}^{-i}, 0)^2 + \min(D_{ijk}^{+i}, 0)^2] = s(x)
/// \end{equation}
///
/// where $D_{ijk}^{\pm i}$ are the backward and forward derivatives in the
/// dimension $i$, and are computed from the stencil
///
/// \tparam Stencil The stencil to use for the scheme.
template <typename Stencil>
struct GodunovUpwind : public Solvable<GodunovUpwind<Stencil>> {
 private:
  /// Defines the type of the stencil for the upwind sheme.
  using stencil_t = std::decay_t<Stencil>;

 public:
  /// Returns the width of the scheme.
  fluidity_host_device constexpr auto width() const {
    return stencil_t().width();
  }

  /// Computes the forward gradient term for the upwinding, which is:
  ///
  /// $ \nabla+(\phi) = \sqrt(
  ///     \sum_i^D \max(\Delta_i^-(\phi),0)^2 + \min(\Delta_i^+(\phi, 0)^2))$
  ///
  /// where $\Delta_i^{\pm}$ are approximations to the first derivatives in the
  /// forward and backward direction for dimension i and are computed by the
  /// stencil.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] dh       The delta for the stencil.
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename Iterator, typename T, typename... Args>
  fluidity_host_device auto forward(
    Iterator&& it  ,
    T          dh  ,
    Args&&...  args
  ) const -> std::decay_t<T> {
    using it_t          = std::decay_t<Iterator>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = decltype(
      stencil_t{}.forward_deriv(it, dh, dim_x, args...)
    );

    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim) {
      const auto back  = std::max(
        stencil.backward_deriv(it, dh, dim, args...), 
        zero
      );
      const auto fwrd  = std::min(
        stencil.forward_deriv(it, dh, dim, args...), 
        zero
      );
      result += std::max(back * back, fwrd * fwrd);
    });
    return std::sqrt(result);
  }

  /// Computes the backward gradient term for the upwinding, which is:
  ///
  /// $ \nabla+(\phi) = \sqrt(
  ///     \sum_i^D \min(\Delta_i^-(\phi),0)^2 + \max(\Delta_i^+(\phi, 0)^2))$
  ///
  /// where $\Delta_i^{\pm}$ are approximations to the first derivatives in the
  /// forward and backward direction for dimension i.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] dh       The delta for the stencil.
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename Iterator, typename T, typename... Args>
  fluidity_host_device auto backward(
    Iterator&& it  , 
    T          dh  , 
    Args&&...  args
  ) const -> std::decay_t<T> {
    using it_t          = std::decay_t<Iterator>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = decltype(
      stencil_t{}.forward_deriv(it, dh, std::size_t{0}, args...)
    );
    
    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim) {
      const auto back  = std::min(
        stencil.backward_deriv(it, dh, dim, args...), 
        zero
      );
      const auto fwrd  = std::max(
        stencil.forward_deriv(it, dh, dim, args...), 
        zero
      );
      result += std::max(back * back, fwrd * fwrd);
    });
    return std::sqrt(result);
  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_HPP
