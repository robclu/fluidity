//==--- fluidity/algorithm/for_each.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  for_each.hpp
/// \brief This file provides for_each wrappers for different containers.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FOR_EACH_HPP
#define FLUIDITY_ALGORITHM_FOR_EACH_HPP

#include <tuple>

namespace fluid {

/// Applies a functor \p functor to each element of the tuple \p tup.
/// \param[in] tup      The tuple to apply the functor to.
/// \param[in] functor  The functor to apply to each element of the tuple.
/// \param[in] args     Additional arguments to the functor.
/// \tparam    Functor  The type of the functor.
/// \tparam    Ts       The types in the tuple.
/// \tparam    Args     The types of the arguments.
template <typename Functor, typename... Ts, typename... Args>
void for_each(std::tuple<Ts...>& tup, Functor&& functor, Args&&... args)
{
  constexpr auto elements = sizeof...(Ts);
  unrolled_for<elements>([&] (auto i)
  {
    auto& t_element = std::get<i>(tup);
    functor(t_element, std::forward<Args>(args)...);
  });
}

/// Applies a functor \p functor to each element of the tuple \p tup.
/// \param[in] tup      The tuple to apply the functor to.
/// \param[in] functor  The functor to apply to each element of the tuple.
/// \param[in] args     Additional arguments to the functor.
/// \tparam    Functor  The type of the functor.
/// \tparam    Ts       The types in the tuple.
/// \tparam    Args     The types of the arguments.
template <typename Functor, typename... Ts, typename... Args>
void for_each(const std::tuple<Ts...>& tup, Functor&& functor, Args&&... args)
{
  constexpr auto elements = sizeof...(Ts);
  unrolled_for<elements>([&] (auto i)
  {
    const auto& t_element = std::get<i>(tup);
    functor(t_element, std::forward<Args>(args)...);
  });
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FOR_EACH_HPP