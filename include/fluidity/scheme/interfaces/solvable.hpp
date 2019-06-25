//==--- fluidity/scheme/interfaces/solvable.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  solvable.hpp
/// \brief This file defines an interface for an object which is solvable.
///        specifically, given an equation involving some data $x$, this
///        interface allows $x$ to be solved for.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_SOLVABLE_HPP
#define FLUIDITY_SCHEME_INTERFACES_SOLVABLE_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

/// The Solvable class provides an interface to which classes which
/// can solve some equation for some data and return the value which solves
/// the given equation. The equation to be solved is defined by the
/// implementation.
/// \tparam SolvableImpl The implementation of the solvable interface.
template <typename SolvableImpl>
class Solvable {
  /// Defines the type of the solvable implementation.
  using impl_t = SolvableImpl;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl()
  {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const
  {
    return static_cast<const impl_t*>(this);
  }

  /// Returns the width required when solving. The width is the maximum
  /// offset (in any dimension) from a cell to another cell whose data needs to
  /// be used by the cell when solving the equation.
  constexpr auto width() const -> std::size_t
  {
    return impl()->width();
  }

  /// Solves the equation, returning the value of the data which solves the
  /// equation.
  /// \param[in] it   The iterable data to solve.
  /// \param[in] args Additional arguments for the evaluation.
  /// \tparam    It   The type of the data iterator. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    Args The types of the additional arguments.
  template <typename It, typename... Args>
  fluidity_host_device auto solve(It&& it, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>,
      "Iterator for Evaluatable must be a multi-dimensional iterator!");
    return impl()->solve_impl(std::forward<It>(it)       ,
                              std::forward<Args>(args)...);
  }
};

/// Returns true if the type T conforms to the Solvable interface.
/// \tparam T The type to check for conformity to the Solvable interface.
template <typename T>
static constexpr auto is_solvable_v = 
  std::is_base_of<Solvable<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_INTERFACES_SOLVABLE_HPP
