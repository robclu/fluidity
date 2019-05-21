//==--- fluidity/scheme/schemes/upwind.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  upwind.hpp
/// \brief This file defines the implementation of an upwind scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_SCHEMES_UPWIND_HPP
#define FLUIDITY_SCHEME_SCHEMES_UPWIND_HPP

#include "../interfaces/scheme.hpp"

namespace fluid  {
namespace scheme {

/// The Upwind struct computes the Godunov upwind update for data.
/// \tparam Stencil The stencil to use for the scheme.
template <typename Stencil>
struct Upwind : public Scheme<Upwind<Stencil>> {
 private:
  /// Defines the type of the stencil for the upwind sheme.
  using stencil_t = std::decay_t<Stencil>;

 public:
  /// Returns the width of the scheme.
  fluidity_host_device constexpr auto width() const
  {
    return stencil_t().width();
  }

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
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    V        The type of the speed data.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename V, typename... Args>
  fluidity_host_device auto invoke(It&& it, T h, V&& v, Args&&... args) const
  {
    // Note: This implemenation can be done using min and max, rather than the
    //       branch. However, although the branch is costly (but becoming less
    //       so with newer gpu architectures), having to compute the unused
    //       factor is far more costly (tested) (and bloats register use) than
    //       the branch.
    using value_t = std::decay_t<decltype(*v)>;
    return *v * (*v <= value_t{0}
              ? grad_backward(std::forward<It>(it)       ,
                              h                          ,
                              std::forward<Args>(args)...)
              : grad_forward(std::forward<It>(it)       ,
                             h                          ,
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
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename... Args>
  fluidity_host_device auto grad_forward(It&& it, T h, Args&&... args) const
  {
    using it_t          = std::decay_t<It>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = 
      decltype(stencil_t{}.forward_deriv(it, h, std::size_t{0}, args...));

    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = 
        std::max(stencil.backward_deriv(it, h, dim, args...), zero);
      const auto fwrd  =
        std::min(stencil.forward_deriv(it, h, dim, args...), zero);
      result += back * back + fwrd * fwrd;
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
  /// \param[in] args     Additional arguments for the stencil.
  /// \tparam    It       The type of the iterator.
  /// \tparam    T        The type of the delta.
  /// \tparam    Args     The types of the stencil arguments.
  template <typename It, typename T, typename... Args>
  fluidity_host_device auto grad_backward(It&& it, T h, Args&&... args) const
  {
    using it_t          = std::decay_t<It>;
    using value_t       = std::decay_t<T>;
    using stencil_ret_t = 
      decltype(stencil_t{}.forward_deriv(it, h, std::size_t{0}, args...));
    
    constexpr auto zero = value_t{0};
    const auto stencil  = stencil_t{};

    auto result = zero;
    unrolled_for<it_t::dimensions>([&] (auto dim)
    {
      const auto back  = 
        std::min(stencil.backward_deriv(it, h, dim, args...), zero);
      const auto fwrd  =
        std::max(stencil.forward_deriv(it, h, dim, args...), zero);
      result += back * back + fwrd * fwrd;
    });
    return std::sqrt(result);
  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_SCHEMES_UPWIND_HPP
