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
  fluidity_host_device constexpr auto width() const
  {
    return stencil_t().width();
  }

  /// Returns the value of the \p it which would solve the Godunov upwind
  /// discretization given the Stencil template. This overload is for the case
  /// that the source data is zero over the whole domain.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] dh       The delta for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  template <typename It, typename T>
  fluidity_host_device auto solve(It&& it, T dh) const
  {
    return get_quadratic(it, dh).solve();
  }

  /// Returns the value of the \p it which would solve the Godunov upwind
  /// discretization given the Stencil template. This overload is for the case
  /// that the source data is zero over the whole domain.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] dh       The delta for the stencil.
  /// \param[in] f        A functor to apply to the quadratic before solving.
  /// \param[in] args     Arguments for the functor.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    F        The type of the functor.
  /// \tparam    Args     The types of the functor arguments.
  template <typename It, typename T, typename F, typename... Args>
  fluidity_host_device auto solve(It&& it, T dh, F&& f, Args&&... args) const
  {
    auto quad = get_quadratic(it, dh);
    f(quad, std::forward<Args>(args)...);

    // Apply the functor to make any changes to the quadratic before solving:
    f(quad, std::forward<Args>(args)...);
    return quad.solve();
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
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename... Args>
  fluidity_host_device auto forward(It&& it, T dh, Args&&... args) const
  {
    using it_t          = std::decay_t<It>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = 
      decltype(stencil_t{}.forward_deriv(it, dh, std::size_t{0}, args...));

    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = 
        std::max(stencil.backward_deriv(it, dh, dim, args...), zero);
      const auto fwrd  =
        std::min(stencil.forward_deriv(it, dh, dim, args...), zero);
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
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename... Args>
  fluidity_host_device auto backward(It&& it, T dh, Args&&... args) const
  {
    using it_t          = std::decay_t<It>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = 
      decltype(stencil_t{}.forward_deriv(it, dh, std::size_t{0}, args...));
    
    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = 
        std::min(stencil.backward_deriv(it, dh, dim, args...), zero);
      const auto fwrd  =
        std::max(stencil.forward_deriv(it, dh, dim, args...), zero);
      result += std::max(back * back, fwrd * fwrd);
    });
    return std::sqrt(result);
  }

 private:
  /// Gets the quadratic to solve before applying any source term.
  /// \param[in] it The iterable data to apply the stencil to.
  /// \param[in] dh The delta for the stencil.
  /// \tparam    It The type of the iterator.
  /// \tparam    T  The type of the delta.
  fluidity_host_device auto get_quadratic(It&& it, T dh) const
  {
    using it_t           = std::decay_t<It>;
    const auto stencil   = stencil_t();
    const auto quad_zero = Quadratic<T>{T(0), T(0), T(0)};
    auto       quad      = quad_zero;

    // Update the quadratic:
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      if (std::max(stencil.backward_deriv(it, dh, dim) > T(0))
      {
        quad += stencil.quadratic_backward(it, dh, dim);
      }

      if (std::min(stencil.forward_deriv(it, dh, dim) < T(0))
      {
        quad += stencil.quadratic_forward(it, dh, dim);
      }
    });
    return quad;
  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_HPP
