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
/// \tparam Form The form of the limiting.
template <typename Form>
struct Linear {
  /// Defines the type of this class.
  using self_t = Linear;

  /// Defines the number of elements required for limiting.
  static constexpr std::size_t width = 1;

  /// Implementation of the linear limiting functionality.
  /// \param[in]  state  The state iterator to limit.
  /// \tparam     IT    The type of the state iterator.
  /// \tparam     Mat   The type of the material for the system.
  /// \tparam     Dim   The type of the dim.
  template <typename IT, typename Mat, typename Dim>
  fluidity_host_device constexpr auto operator()(IT&& state, Mat&&, Dim) const
  {
    using state_t     = typename std::decay_t<decltype(*state)>;
    using value_t     = typename state_t::value_t;
    using container_t = Array<value_t, state_t::elements>;

    container_t container;
    unrolled_for_bounded<state_t::elements>([&] (auto i)
    {
      container[i] = value_t{0};
    });
    return container;
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_LINEAR_LIMITER_HPP
