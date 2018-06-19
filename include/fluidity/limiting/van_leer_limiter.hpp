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

#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/container/array.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/utility/portability.hpp>
#include <algorithm>

namespace fluid {
namespace limit {

/// The VanLeer class defines a functor which implements VanLeer limiting as
/// per: Toro, page 510, equation 14.54, which is:
/// 
///   \begin{equation}
///     \Eita_{vl}(r) = 
///       \begin{cases}
///         0           &, if r \le 0   \\
///         min\{L, R\} &, if r \ge 0
///       \end{cases}
///   \end{equation}
///
///   where:
///
///    $ L = \frac{2r}{1 + r} $
///    $ R = \Eita_R(r)       $
struct VanLeer {
  /// Defines the type of this class.
  using self_t = VanLeer;

  /// Defines the number of elements required for limiting.
  static constexpr std::size_t width = 2;

  /// Implementation of the limit function which applies the limiting to an
  /// iterator, calling the limit method on each of the iterator elements.
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

    constexpr auto dim   = Dimension<Value>{};
    constexpr auto scale = value_t{0.5};
    
    const auto fwrd_diff = state_it.forward_diff(dim);
    const auto back_diff = state_it.backward_diff(dim);

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
    constexpr auto zero = value_t{0}, one = value_t{1}, two = value_t{2};

    const auto r = left / right;
    return (r <= zero || right == zero) 
           ? zero : two * std::min(r, one) / (one + r);
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_VAN_LEER_LIMITER_HPP
