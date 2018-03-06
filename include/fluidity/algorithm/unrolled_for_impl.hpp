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

#ifndef FLUIDITY_ALGORITHM_UNROLLED_FOR_IMPL_HPP
#define FLUIDITY_ALGORITHM_UNROLLED_FOR_IMPL_HPP

#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace detail {

/// This is a wrapper struct which stores an index which can provide a compile
/// time and runtime value.
/// \tparam   Value  The value of the index.
template <std::size_t Value>
struct Index {
  /// Returns the value of the index.
  fluidity_host_device static constexpr std::size_t value()
  {
    return Value;
  }

  /// Conversion to size_t so that the index can be used exactly as a size type.
  fluidity_host_device constexpr operator size_t()
  {
    return Value;
  }
};

/// The Unroll struct invokes a callable object N times, where the invokations
/// are unrolled at compile time.
/// \tparam Amount  The amount of unrolling to do.
template <std::size_t Amount>
struct Unroll : Unroll<Amount - 1> {
  /// Defines the value for the previous level.
  static constexpr std::size_t previous_level_v = Amount - 1;
  /// Defines the type of the case class which invokes at the previous level.
  using previous_level_t = Unroll<previous_level_v>;

  /// Passes the \p functor and \p args to the previous level to invoke, and
  /// then invokes at this level.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  fluidity_host_device Unroll(Functor&& functor, Args&&... args)
  : previous_level_t(std::forward<Functor>(functor),
                     std::forward<Args>(args)...   )
  {
    functor(Index<previous_level_v>(), std::forward<Args>(args)...);
  }
};

/// Specialization of the unrolling class to terminate the urolling at the
/// lowest level.
template <>
struct Unroll<1> {
  /// Invokes the functor with the given args.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  fluidity_host_device Unroll(Functor&& functor, Args&&... args)
  {
    functor(Index<0>(), std::forward<Args>(args)...);
  }
};

/// Specialization of the unrolling class for the case that 0 unrolling is
/// specified. This specialization does nothing, it's defined for generic
/// code that may invoke it.
template <>
struct Unroll<0> {
  /// Does nothing.
  /// \param[in]  functor   Placeholder for a functor.
  /// \param[in]  args      Placeholder for the functor arguments.
  /// \tparam     Functor   The type of the functor.
  /// \tparam     Args      The type of the arguments.
  template <typename Functor, typename... Args>
  fluidity_host_device 
  Unroll(Functor&& /*functor*/, Args&&... /*args*/) {}
};

} // namespace detail
} // namespace fluid

#endif // FLUIDITY_ALGORITHM_UNROLLED_FOR_HPP