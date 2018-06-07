//==--- fluidity/limiting/linear_limiter.hpp --------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  linear_limiter.hpp
/// \brief This file defines an implementation of a limiter which performs first
///        order limiting.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_LINEAR_LIMITER_HPP
#define FLUIDITY_LIMITING_LINEAR_LIMITER_HPP

#include <fluidity/container/array.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/utility/portability.hpp>
#include <type_traits>

namespace fluid {
namespace limit {

/// The Linear limiter class defines a functor which performs linear limiting,
/// as per:
/// 
///   Toro, page 506, equation 14.37, w = 0
struct Linear {
  /// Defines the type of this class.
  using self_t = Linear;

  /// Defines the number of elements required for limiting.
  static constexpr std::size_t width = 1;

  /// Implementation of the linear limiting functionality.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  dim       The (spacial) dimension to limit over.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Value     The value which defines the dimension.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device constexpr auto
  operator()(Iterator&& state_it, Dimension<Value> /*dim*/) const
  {
    using state_t     = typename std::decay_t<decltype(*state_it)>;
    using value_t     = typename state_t::value_t;
    using container_t = Array<value_t, state_t::elements>;

    container_t container;
    unrolled_for<state_t::elements>([&] (auto i)
    {
      container[i] = value_t{0};
    });
    return container;
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_LINEAR_LIMITER_HPP
