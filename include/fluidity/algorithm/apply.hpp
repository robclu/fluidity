//==--- fluidity/algorithm/apply.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  apply.hpp
/// \brief This file defines functionality to apply a function to a pack of
///        arguments.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_APPLY_HPP
#define FLUIDITY_ALGORITHM_APPLY_HPP

#include "for_each.hpp"

namespace fluid {

/// Applies the \p functor to each of the \p objs objects. This is simply a
/// utility function for the case that the same operation needs to be performed
/// on a number of objects which are not stored in an iterable container.
///
/// Consider that there are a number of iterators, and each iterator needs to be
/// offset in each direction, then this may be performed with apply as follows:
///
/// \code{.cpp}
/// auto it1, it2, it3, it4 = make_iterators();
///
/// for (auto dim : range(it1.dimensions))
/// {
///   apply([] (auto& it)
///   {
///     it.shift(it, dim);
///   }, it1, it2, it3, it4);
/// }
/// \endcode
///
/// \param[in] functor The functor to apply to each object.
/// \param[in] objs    The objects to apply the functor to.
/// \tparam    F       The type of the functor to apply.
/// \tparam    Objs    The types of the objects.
template <typename F, typename... Objs>
fluidity_host_device constexpr inline auto apply(F&& functor, Objs&&... objs)
{
  auto t = make_tuple(std::forward<Objs>(objs)...);
  unrolled_for<sizeof...(Objs)>([&] (auto i) 
  {
    functor(get<i>(t));
  });
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_APPLY_HPP
