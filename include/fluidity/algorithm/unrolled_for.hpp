//==--- fluidity/algorithm/unrolled_for.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unrolled_for.hpp
/// \brief This file defines the implementation of a function with allows the
///        compile time unrolling of a function body to execute N times.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_UNROLLED_FOR_HPP
#define FLUIDITY_ALGORITHM_UNROLLED_FOR_HPP

#include "if_constexpr.hpp"
#include "unrolled_for_impl.hpp"
#include <fluidity/iterator/range.hpp>

namespace fluid {

/// Applies the functor Amount times, and passes the index of the unrolling
/// as the first argument to the functor. The unrolling is performed at compile
/// time, so the number of iterations should not be too large. The index
/// parameter has a compile-time value, and can therefore be used in constexpr
/// contexts. For example:
/// 
/// ~~~cpp
/// auto tuple = std::make_tuple("string", 4, AType*());
/// unrolled_for<2>([&tuple] (auto i) {
///   do_something(get<i>(tuple), get<i + 1>(tuple)); 
/// });
/// ~~~
/// 
/// Which effectively will become:
/// 
/// ~~~cpp
/// do_something(get<0>(tuple), get<1>(tuple)); 
/// do_something(get<1>(tuple), get<2>(tuple)); 
/// ~~~
/// 
/// \note If you want to ensure that the unrolling is bounded, then use
///       `unrolled_for_bounded`. See below for details.
/// 
/// \param[in]  functor   The functor to unroll.
/// \param[in]  args      The arguments to the functor.
/// \tparam     Amount    The amount of unrolling to do.
/// \tparam     Functor   The type of the functor.
/// \tparam     Args      The type of the functor arguments.
template <std::size_t Amount, typename Functor, typename... Args>
fluidity_host_device constexpr inline auto
unrolled_for(Functor&& functor, Args&&... args)
{
  // TODO: Check if this returns a type ...
  detail::Unroll<Amount> unrolled(std::forward<Functor>(functor), 
                                  std::forward<Args>(args)...  );
}

/// Applies the functor Amount times. However, this is a bounded version and
/// is safer than `unrolled_for`` in that it will not unroll if the value of
/// Amount is larger than the values defined by MAX_UNROLL_DEPTH at compile
/// time. If Amount is larger than MAX_UNROLL_DEPTH then a normal loop is
/// invoked on the functor. In the case that Amount <= MAX_UNROLL_DEPTH, then
/// this behaves exactly as `unrolled_for`. For example:
/// 
/// ~~~cpp
/// auto tuple = std::make_tuple("string", 4, AType*());
/// unrolled_for<2>([&tuple] (auto i) {
///   do_something(get<i>(tuple), get<i + 1>(tuple)); 
/// });
/// ~~~
/// 
/// Which effectively will become:
/// 
/// ~~~cpp
/// do_something(get<0>(tuple), get<1>(tuple)); 
/// do_something(get<1>(tuple), get<2>(tuple)); 
/// ~~~
/// 
/// \param[in]  functor   The functor to unroll.
/// \param[in]  args      The arguments to the functor.
/// \tparam     Amount    The amount of unrolling to do.
/// \tparam     Functor   The type of the functor.
/// \tparam     Args      The type of the functor arguments.
template <std::size_t Amount, typename Functor, typename... Args>
fluidity_host_device constexpr inline auto
unrolled_for_bounded(Functor&& functor, Args&&... args)
{
  if_constexpr<Amount <= max_unroll_depth>([&]
  {
    unrolled_for<Amount>(std::forward<Functor>(functor),
                         std::forward<Args>(args)...   );
  }, [&]
  {
    for (const auto i : range(Amount))
    {
      functor(i, std::forward<Args>(args)...);
    }
  });
}


} // namespace fluid

#endif // FLUIDITY_ALGORITHM_UNROLLED_FOR_HPP
