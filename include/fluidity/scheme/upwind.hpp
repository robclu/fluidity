//==--- fluidity/scheme/upwind.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  upwind.hpp
/// \brief This file defines the implementation of an upwind scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_UPWIND_HPP
#define FLUIDITY_SCHEME_UPWIND_HPP

#include "scheme.hpp"

namespace fluid  {
namespace scheme {


struct Upwind : public Scheme<Upwind> {
  /// Implementation of the upwind scheme, which is defined as the following:
  /// 
  /// $ H(\phi, v) = max(v, 0) \nabla^-(\phi) + min(v, 0) \nabla^+(\phi) $
  ///
  /// where the $\nabla^{\pm} terms are:
  ///
  /// $ \nabla^{pm}(\phi) = \sqrt(\sum_i^D (max(D^-, 0)^2 + min(D^+, 0))) $
  ///
  /// and where the $D^{\pm}$ terms are approximations to the first derivates of 
  /// the \p it iterable data, and can be computed using the \p stencil.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] v        The data which defines the speed of the scheme.
  /// \param[in] h        The delta for the stencil.
  /// \param[in] stencil  The stencil to use to compute the derivatives.
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    V        The type of the speed data.
  /// \tparam    T        The type of the delta.
  /// \tparam    S        The type of the stencil.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename V, typename T, typename S, typename... Args>
  fluidity_host_device auto
  invoke(It&& it, V&& v, T h, S&& stencil, Args&&... args) const
  {
    // Note: This implemenation can be done using min and max, rather than the
    //       branch. However, although the branch is costly (but becoming less
    //       so with newer gpu architectures), having to compute the unused
    //       factor is far more costly (tested) (and bloats register use) than
    //       the branch.
    using value_t = std::decay_t<decltype(*v)>;
    return *v * (*v <= value_t{0}
              ? grad_backward(std::forward<It>(it)      ,
                              h                         ,
                              std::forward<S>(stencil)  ,
                              std::forwrd<Args>(args)...)
              : grad_forward(std::forward<It>(it)       ,
                             h                          ,
                             std::forward<S>(stencil)   ,
                             std::forward<Args>(args)...));
  }

 private:
  /// Computes the forward gradient term for the upwinding, which is:
  ///
  /// $ \nabla+(\phi) = \sqrt(
  ///     \sum_i^D \max(\Delta_i^-(\phi),0)^2 + \min(\Delta_i^+(\phi, 0)^2))$
  ///
  /// where $\Delta_i^{\pm}$ are approximations to the first derivatives in the
  /// forward and backward direction for dimension i.
  ///
  /// \param[in] it       The iterable data to apply the stencil to.
  /// \param[in] h        The delta for the stencil.
  /// \param[in] stencil  The stencil to use to compute the derivatives.
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    S        The type of the stencil.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename S, typename... Args>
  fluidity_host_device auto
  grad_forward(It&& it, T h, S&& stencil, Args&&... args) const
  {
    using stencil_ret_t = decltype(stencil.forward_deriv(it, h, args...));
    using value_t       = std::decay_t<stencil_ret_t>;
    using it_t          = std::decay_t<it>;
    constexpr auto zero = value_t{0};
    auto result         = zero;

    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = std::max(stencil.backward_deriv(it, h, args...), zero);
      const auto fwrd  = std::min(stencil.forward_deriv(it, h, args...), zero);
      result          += back * back + fwrd * fwrd;
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
  /// \param[in] h        The delta for the stencil.
  /// \param[in] stencil  The stencil to use to compute the derivatives.
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    S        The type of the stencil.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename Stencil, typename... Args>
  fluidity_host_device auto
  grad_backward(It&& it, T h, Stencil&& stencil, Args&&... args) const
  {
    using stencil_ret_t = decltype(stencil.forward_deriv(it, h, args...));
    using value_t       = std::decay_t<stencil_ret_t>;
    using it_t          = std::decay_t<it>;
    constexpr auto zero = value_t{0};
    auto result         = zero;

    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = std::min(stencil.backward_deriv(it, h, args...), zero);
      const auto fwrd  = std::max(stencil.forward_deriv(it, h, args...), zero);
      result          += back * back + fwrd * fwrd;
    });
    return std::sqrt(result);
  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_UPWIND_HPP
#define FLUIDITY_LEVELSET_FIRST_ORDER_EVOLUTION_HPP
