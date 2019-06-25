//==--- fluidity/limiting/superbee_limiter_unsplit.hpp ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  superbee_limiter_unsplit.hpp
/// \brief This file defines an implementation of a limiter which performs
///        limiting using the SUPERBEE method, speficially for an unsplit flux
///        method. 
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_SUPERBEE_LIMITER_UNSPLIT_HPP
#define FLUIDITY_LIMITING_SUPERBEE_LIMITER_UNSPLIT_HPP

#include "limiter.hpp"
#include <algorithm>

namespace fluid {
namespace limit {

/// The Superbee limiter class defines a functor which performs superbee
/// limiting as per: Toro, page 510, equation 14.53, which is:
///  
/// \begin{equation}
///   \Eita_{sb}(r) = 
///     \begin{cases}
///       0                       &, if r \le 0                   \\
///       2r                      &, if 0 \le r \le \frac{1}{2}   \\
///       1                       &, if \frac{1}{2} \le r \le 1   \\
///       min\{r, \Eita_R(r), 2\} &, if r \ge 1
///     \end{cases}
/// \end{equation}
///
/// \tparam Form The form of the variables to limit on.
template <typename Form>
struct SuperbeeUnsplit : public Limiter<SuperbeeUnsplit<Form>> {
  /// Defines the form of the limiting.
  using form_t = prim_form_t;

  /// Implementation of the limit function which applies the limiting to an
  /// iterator, calling the limit method on each of the iterator elements.
  /// \param[in]  state   The state iterator to limit.
  /// \param[in]  mat     The material for the system.
  /// \tparam     IT      The type of the state iterator.
  /// \tparam     Mat     The type of the material.
  /// \tparam     Dim     The tyoe of the dimension.
  template <typename IT, typename Mat, typename Dim>
  fluidity_host_device constexpr auto
  limit_impl(IT&& state, Mat&& mat, Dim dim) const
  {
    using state_t = std::decay_t<decltype(state->primitive(mat))>;
    using value_t = typename state_t::value_t;

    const auto fwrd_diff = forward_diff<form_t>(state, mat, dim);
    const auto back_diff = backward_diff<form_t>(state, mat, dim);
    auto cent_diff = value_t{0.5} * (fwrd_diff + back_diff);

    unrolled_for<state_t::elements>([&] (auto i)
    {
      cent_diff[i] *= limit_single(back_diff[i], fwrd_diff[i]);
    });
    return cent_diff;
  }

 private:
  /// Defines the type of the base class.
  using base_t = Limiter<SuperbeeUnsplit<Form>>;

  /// Allow the base class to use the implementation details.
  friend base_t;

  /// Returns the limited value of a single element, as defined in the class
  /// description. 
  /// \param[in] left    The left state to limit on.
  /// \param[in] right   The right state to limit on.
  template <typename T>
  fluidity_host_device constexpr auto limit_single(T&& left, T&& right) const
  {
    using value_t = std::decay_t<T>;
    constexpr auto zero = value_t{0}  ,
                   half = value_t{0.5},
                   one  = value_t{1}  ,
                   two  = value_t{2}  ;

    const auto r = left / right;

    if      (r <= zero || right == zero) { return zero;           }
    else if (r <= half)                  { return value_t{2} * r; }
    else if (r <= one)                   { return one;            }

    const auto Er = two / (one + r);
    return std::min(Er, std::min(r, two));
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_LINEAR_LIMITER_HPP