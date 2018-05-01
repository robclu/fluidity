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
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/utility/portability.hpp>
#include <algorithm>

namespace fluid {
namespace limit {

/// The VanLeer class defines a functor which implements VanLeer limiting.
struct VanLeer {
  /// Defines the type of this class.
  using self_t = VanLeer;

  /// Defines the number of elements required for limiting.
  static constexpr std::size_t width = 2;

  /// Returns the limited value of a single element, defined as follows:
  /// limiting is defined as:
  /// 
  ///   \begin{equation}
  ///     \Eita = 
  ///       \begin{cases}
  ///         \textrm{min}(2|\alpha_L|, 2|\alpha_R|, \alpha_C)
  ///         0
  ///       \end{cases}
  ///   \end{equation}
  ///   
  /// where $\alpha_{L,R,C}$ are the backward, forward, and central differences.
  /// 
  /// \param[in] central The central state to limit on.
  /// \param[in] left    The left state to limit on.
  /// \param[in] right   The right state to limit on.
  template <typename T>
  fluidity_host_device constexpr decltype(auto)
  operator()(T&& central, T&& left, T&& right) const 
  {
    using value_t = std::decay_t<T>;
    return (left * right < value_t{0})
           ? value_t{0}
           : math::signum(central)
           * std::min(value_t{0.5} * std::abs(central),
                      value_t{2.0} * std::min(std::abs(left), std::abs(right)));
  }

  /// Implementation of the limit function which applies the limiting to an
  /// iterator, calling the limit method on each of the iterator elements.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  dim       The (spacial) dimension to limit over.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Value     The value which defines the dimension.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device constexpr decltype(auto)
  operator()(Iterator&& state_it, Dimension<Value> /*dim*/) const
  {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = typename state_t::value_t;

    Array<value_t, state_t::elements> limited;
    unrolled_for<state_t::elements>([&] (auto i)
    {
      constexpr auto limiter = self_t{};
      constexpr auto dim     = Dimension<Value>{};

      limited[i] = limiter(state_it.central_diff(dim)[i] ,
                           state_it.backward_diff(dim)[i],
                           state_it.forward_diff(dim)[i] );
    });
    return limited;
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_VAN_LEER_LIMITER_HPP
