//==--- fluidity/limiting/van_leer_limiter.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  van_leer_limiter.hpp
/// \brief This file defines an implementation of the Van Leer limiting.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_VAN_LEER_LIMITER_HPP
#define FLUIDITY_LIMITING_VAN_LEER_LIMITER_HPP

#include "limiter.hpp"
#include <algorithm>

namespace fluid {
namespace limit {

/// The VanLeer class defines a functor which implements VanLeer limiting as
/// per: Toro, page 510, equation 14.54, which is:
/// 
/// \begin{equation}
///   \Eita_{vl}(r) = 
///     \begin{cases}
///       0           &, if r \le 0   \\
///       min\{L, R\} &, if r \ge 0
///     \end{cases}
/// \end{equation}
/// where:
///  $ L = \frac{2r}{1 + r} $
///  $ R = \Eita_R(r)       $
///
/// \tparam Form The form of the state variables to limit on.
template <typename Form>
struct VanLeer : public Limiter<VanLeer<Form>> {
  /// Defines the form of the variables to limit on.
  using form_t = Form;

  /// Implementation of the limit function which applies the limiting to an
  /// iterator, calling the limit method on each of the iterator elements.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  material  The material for the system.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Material  The type of the material.
  /// \tparam     Value     The value which defines the dimension.
  template <typename Iterator, typename Material, typename Dim>
  fluidity_host_device constexpr auto
  limit_impl(Iterator&& state_it, Material&& mat, Dim dim) const
  {
    return this->limit_generic(std::forward<Iterator>(state_it),
                               std::forward<Material>(mat)     ,
                               dim                             );
  }

  /// Returns the limited value of a single element, as defined in the class
  /// description. 
  /// \param[in] left    The left state to limit on.
  /// \param[in] right   The right state to limit on.
  template <typename T>
  fluidity_host_device constexpr auto limit_single(T&& left, T&& right) const
  {
    using value_t = std::decay_t<T>;
    constexpr auto zero = value_t{0}, one = value_t{1}, two = value_t{2};

    const auto r = left / right;
    return (r <= zero || right == zero) 
           ? zero : two * std::min(r, one) / (one + r);
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_VAN_LEER_LIMITER_HPP
