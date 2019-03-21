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

#include "for_each.cuh"
#include <fluidity/container/tuple.hpp>
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


/// Applies a functor \p functor to each element of the tuple \p tup.
/// \param[in] tup      The tuple to apply the functor to.
/// \param[in] functor  The functor to apply to each element of the tuple.
/// \param[in] args     Additional arguments to the functor.
/// \tparam    Functor  The type of the functor.
/// \tparam    Ts       The types in the tuple.
/// \tparam    Args     The types of the arguments.
template <typename Functor, typename... Ts, typename... Args>
void for_each(Tuple<Ts...>& tup, Functor&& functor, Args&&... args)
{
  constexpr auto elements = sizeof...(Ts);
  unrolled_for<elements>([&] (auto i)
  {
    auto& t_element = get<i>(tup);
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
void for_each(const Tuple<Ts...>& tup, Functor&& functor, Args&&... args)
{
  constexpr auto elements = sizeof...(Ts);
  unrolled_for<elements>([&] (auto i)
  {
    const auto& t_element = get<i>(tup);
    functor(t_element, std::forward<Args>(args)...);
  });
}

/// Fills the multi dimensional iterator using \p pred to set value of the
/// elements. The \p pred must be a callable predicate, whose first argument is
/// the multidimensional iterator which points to the element to set. The
/// signature of the function is the following:
///
/// \begin{code}
/// void predicate(iter_t& iter, Args... args)
/// {
///   // Set the element:
///   *iter = value;
/// }
/// \endcode
///
/// This signature allows the iterator functionality to be used to allow for
/// more complex filling techniques.
///
/// This overload will only be enabled when the Iterator is multi dimensional.
///
/// \param[in] iter     The iterator to start filling from.
/// \param[in] pred     The predicate to use to set the value.
/// \param[in] args     Additional arguments for the predicate.
/// \tparam    Iterator The type of the iterator.
/// \tparam    Pred     The type of the predicate.
/// \tparam    Args     The type of arguments for a callable predicate.
template <typename Iterator               ,
          typename Pred                   ,
          multiit_enable_t<Iterator> = 0  ,
          exec::gpu_enable_t<Iterator> = 0>
void for_each(Iterator&& it, Pred&& pred)
{
  detail::cuda::for_each(std::forward<Iterator>(it), std::forward<Pred>(pred));
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FOR_EACH_HPP