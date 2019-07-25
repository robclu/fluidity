//==--- fluidity/scheme/interfaces/evaluatable.hpp --------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  evaluatable.hpp
/// \brief This file defines an interface for a class which can be evaluated.
///        That is, it takes some data, and evaluates it to compute a result
///        based on the data. Examples include Hamiltonians and Flux functions.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_EVALUATABLE_HPP
#define FLUIDITY_SCHEME_INTERFACES_EVALUATABLE_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

/// The Evaluatable class provides an interface to which classes which
/// can evaluate some data and return the result of the evaluation, must
/// conform. The specific method of the evaluation is implementation dependent.
/// \tparam EvaluatableImpl The implementation of the Evaluatable interface.
template <typename EvaluatableImpl>
class Evaluatable {
  /// Defines the type of the evaluatable implementation.
  using impl_t = EvaluatableImpl;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl() {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const {
    return static_cast<const impl_t*>(this);
  }

  /// Returns the width required when evaluating. The width is the maximum
  /// offset (in any dimension) from a cell to another cell whose data needs to
  /// be used by the cell when performing the evaluation.
  constexpr auto width() const -> std::size_t {
    return impl()->width();
  }

  /// This method evaluates the data, and returns the computed value. This
  /// overload is only enabled when the \p v_it is a multi-dimensional iterator.
  /// \param[in] it   The iterable data to evaluate.
  /// \param[in] v_it Iterable data to multiply with the iterable data.
  /// \param[in] dh   The discretization which the evaluation can use.
  /// \param[in] args Additional arguments for the evaluation.
  /// \tparam    It   The type of the data iterator. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    VIt  The type of the multiplication data. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    T    The type of the discretization value. It must be possible
  ///                 to perform mathematical operations with this type and the
  ///                 data iterated over.
  /// \tparam    Args The types of the additional arguments.
  template <typename    It  ,
            typename    VIt ,
            typename    T   ,
            typename... Args, multiit_enable_t<VIt> = 0>
  fluidity_host_device auto
  evaluate(It&& it, VIt&& v_it, T dh, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>,
      "Iterator for Evaluatable must be a multi-dimensional iterator!");
    return impl()->evaluate_impl(std::forward<It>(it)       ,
                                 std::forward<VIt>(v_it)    ,
                                 dh                         ,
                                 std::forward<Args>(args)...);
  }

  /// This method evaluates the data, and returns the computed value. This
  /// overload is only enabled when the \p f is not a multi-dimensional
  /// iterator.
  /// 
  /// \todo Add trait check that \p f is callable.
  ///
  /// \param[in] it   The iterable data to evaluate.
  /// \param[in] f    A functor to apply to the data.
  /// \param[in] dh   The discretization which the evaluation can use.
  /// \param[in] args Additional arguments for the evaluation.
  /// \tparam    It   The type of the data iterator. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    F    The type of the functor.
  /// \tparam    T    The type of the discretization value. It must be possible
  ///                 to perform mathematical operations with this type and the
  ///                 data iterated over.
  /// \tparam    Args The types of the additional arguments.
  template <typename    It  ,
            typename    F   ,
            typename    T   ,
            typename... Args, nonmultiit_enable_t<F> = 0>
  fluidity_host_device auto
  evaluate(It&& it, F&& f, T dh, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>,
      "Iterator for Evaluatable must be a multi-dimensional iterator!");
    return impl()->evaluate_impl(std::forward<It>(it)       ,
                                 std::forward<F>(f)         ,
                                 dh                         ,
                                 std::forward<Args>(args)...);
  }
};

/// Returns true if the type T conforms to the Evaluatable interface.
/// \tparam T The type to check for conformity to the Evaluatable inteface.
template <typename T>
static constexpr auto is_evaluatable_v = 
  std::is_base_of<Evaluatable<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_INTERFACES_EVALUATABLE_HPP

