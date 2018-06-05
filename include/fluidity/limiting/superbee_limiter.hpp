//==--- fluidity/limiting/superbee_limiter.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  superbee_limiter.hpp
/// \brief This file defines an implementation of a limiter which performs
///        limiting using the SUPERBEE method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_SUPERBEE_LIMITER_HPP
#define FLUIDITY_LIMITING_SUPERBEE_LIMITER_HPP

#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/container/array.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/utility/portability.hpp>
#include <algorithm>

namespace fluid {
namespace limit {

/// The Superbee limiter class defines a functor which performs superbee
/// limiting as per: Toro, page 510, equation 14.53, which is:
///  
///   \Eita_{sb}(r) = 
///     \begin{cases}
///       0                       &, if r \le 0                   \\
///       2r                      &, if 0 \le r \le \frac{1}{2}   \\
///       1                       &, if \frac{1}{2} \le r \le 1   \\
///       min\{r, \Eita_R(r), 2\} &, if r \ge 1
///     \end{cases}
struct Superbee {
  /// Defines the type of this class.
  using self_t = Superbee;

  /// Defines the number of elements (on one side) for limiting.
  static constexpr std::size_t width = 2;

  /// Implementation of the SUPERBEE limiting functionality.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  dim       The (spacial) dimension to limit over.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Value     The value which defines the dimension.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device constexpr auto
  operator()(Iterator&& state_it, Dimension<Value> /*dim*/) const
  {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = typename state_t::value_t;
    Array<value_t, state_t::elements> limited;

    constexpr auto dim   = Dimension<Value>();
    constexpr auto scale = value_t{0.5};

    const auto fwrd_diff = state_it.forward_diff(dim);
    const auto back_diff = state_it.backward_diff(dim);

    // Limit each of the variables:
    unrolled_for<state_t::elements>([&] (auto i)
    {
      limited[i] = scale 
                 * limit_single(back_diff[i], fwrd_diff[i])
                 * (back_diff[i] + fwrd_diff[i]);
    });
    return limited;
  }

 private:
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