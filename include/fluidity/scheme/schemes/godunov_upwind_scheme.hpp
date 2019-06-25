//==--- fluidity/scheme/schemes/godunov_upwind_solver.hpp -- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  godunov_upwind_scheme.hpp
/// \brief This file defines the implementation of a first order
///        Godunov upwind scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_SCHEME_HPP
#define FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_SCHEME_HPP

#include <fluidity/algorithm/small_sort.hpp>
#include <fluidity/traits/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

struct GodunovUpwindScheme {
  template <typename I,
            typename T,
            typename F, traits::multiit_enable_t<I> = 0>
  fluidity_host_device auto solve(I&& it, T dh, F&& f) const
      -> typename std::decay_t<I>::value_t {
    return solve_impl(std::forward<I>(it), dh, std::forward<F>(f));
  }

 private:
  template <typename I, typename T, typename F, traits::it_1d_enable_t<I> = 0>
  fluidity_host_device auto solve_impl(I&& it, T dh, F&& f) const {
    return std::min(*it.offset(-1, dim_x), *it.offset(1, dim_x)) + (f * dh);
  }

  template <typename I, typename T, typename F, traits::it_2d_enable_t<I> = 0>
  fluidity_host_device auto solve_impl(I&& it, T dh, F&& f) const {
    const auto a   = std::min(*it.offset(-1, dim_x), *it.offset(1, dim_x));
    const auto b   = std::min(*it.offset(-1, dim_y), *it.offset(1, dim_y));
    const auto fh  = f * dh;
    const auto amb = a - b;

    return std::abs(amb) >= fh
      ? std::min(a, b) + fh
      : T(0.5) * (a + b + std::sqrt(T(2) * fh * fh - (amb * amb)));
  }

  template <typename I, typename T, typename F, traits::it_3d_enable_t<I> = 0>
  fluidity_host_device auto solve_impl(I&& it, T dh, F&& f) const {
    auto v = Array<T, 3>(
      std::min(*it.offset(-1, dim_x), *it.offset(1, dim_x)),
      std::min(*it.offset(-1, dim_y), *it.offset(1, dim_y)),
      std::min(*it.offset(-1, dim_z), *it.offset(1, dim_z))
    );

    const auto fh = f * dh;
    auto r        = v[0] + fh;
    if (r <= v[1]) { return r; }

    r = v[0] - v[1];
    r = T(0.5) * (v[0] + v[1] + std::sqrt(T(2) * fh * fh - (r * r)));
    if (r <= v[2]) { return r; }

    r = v[0] + v[1] + v[2];
    constexpr auto factor = T(1) / T(6);
    return factor * (
      T(2) * r + 
      std::sqrt(
        T(4) * r * r - 
        T(12) * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - fh * fh)
      )
    );
  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_SCHEMES_GODUNOV_UPWIND_SCHEME_HPP

