//==--- fluidity/levelset/first_order_evolution.hpp -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  first_order_evolution.hpp
/// \brief This file defines functionality for evolving the levelset.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_FIRST_ORDER_EVOLUTION_HPP
#define FLUIDITY_LEVELSET_FIRST_ORDER_EVOLUTION_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace levelset {

struct FirstOrderUpwind {

  template <typename It, typename V, typename T>
  fluidity_host_device static void evolve(It&& in, It&& out, V&& v, T dt)
  {
    using it_t          = std::decay_t<It>;
    using value_t       = typename it_t::value_t;
    constexpr auto dims = it_t::dimensions;

    // Offset the iterators ...
    unrolled_for<dims>([&] (auto dim)
    {
      in.shift(flattened_id(dim), dim);
      out.shift(flattened_id(dim), dim);
      v.shift(flattened_id(dim), dim);
    });

    auto factor_fwrd = value_t{0};
    auto factor_back = value_t{0};
    unrolled_for<dims>([&] (auto dim)
    {
      auto back_diff = in.backward_diff(dim);
      auto fwrd_diff = in.forward_diff(dim);

      factor_fwrd +=
        std::pow(std::max(back_diff, value_t{0}), 2) +
        std::pow(std::min(fwrd_diff, value_t{0}), 2);
      factor_back +=
        std::pow(std::min(back_diff, value_t{0}), 2) +
        std::pow(std::max(fwrd_diff, value_t{0}), 2);
    });

    factor_fwrd = std::max(*v, value_t{0}) * std::sqrt(factor_fwrd);
    factor_back = std::min(*v, value_t{0}) * std::sqrt(factor_back);

    //printf("factor : %6.4f\n", dt * (factor_fwrd + factor_back));

    *out = *in + dt * (factor_fwrd + factor_back);
  }
};

}} // namespace fluid::levelset


#endif // FLUIDITY_LEVELSET_FIRST_ORDER_EVOLUTION_HPP
