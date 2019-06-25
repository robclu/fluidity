//==--- fluidity/limiting/van_leer_limiter_unsplit.hpp ----- -*- C++ -*- ---==//
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

#ifndef FLUIDITY_LIMITING_VAN_LEER_LIMITER_UNSPLIT_HPP
#define FLUIDITY_LIMITING_VAN_LEER_LIMITER_UNSPLIT_HPP

#include "limiter.hpp"
#include <algorithm>

namespace fluid {
namespace limit {

/// The VanLeer class defines a functor which implements VanLeer limiting
/// specifically for an unsplit flux update.

///
/// \tparam Form The form of the state variables to limit on.
template <typename Form>
struct VanLeerUnsplit : public Limiter<VanLeerUnsplit<Form>> {
  /// Defines the form of the variables to limit on. For this limiter, we always
  /// limit on primitive variables.
  using form_t = prim_form_t;

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
    using state_t = std::decay_t<decltype(state_it->primitive(mat))>;
    using value_t = typename state_t::value_t;
    
    const auto fwrd_diff = forward_diff<form_t>(state_it, mat, dim);
    const auto back_diff = backward_diff<form_t>(state_it, mat, dim);
    auto cent_diff = value_t{0.5} * (back_diff + fwrd_diff);

    unrolled_for<state_t::elements>([&] (auto i)
    {
      cent_diff[i] *= limit_single(back_diff[i], fwrd_diff[i]);
    });
    return cent_diff;
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