//==--- fluidity/scheme/eikonal/fast_iterative_method.hpp -- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fast_iterative_method.hpp
/// \brief This file defines an implementation of the Fast Iterative Method of
///        \cite Jeong2008.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_EIKONAL_FAST_ITERATIVE_METHOD_HPP
#define FLUIDITY_SCHEME_EIKONAL_FAST_ITERATIVE_METHOD_HPP

#include "../cuda/fast_iterative_method.cuh"
#include "../interfaces/eikonal.hpp"
#include <fluidity/traits/device_traits.hpp>

namespace fluid   {
namespace scheme  {
namespace eikonal {

/// The FastIterative class implements the Eikonal interface using the Fast
/// Iterative Method (FIM) of \cite Jeong2008.
/// 
/// This could be generalized to take a template which conforms to the Scheme
/// interface which computes the Hamiltonaian, however, since the 
class FastIterative : public Eikonal<FastIterative> {
 public:
  /// Returns the width required by the Eikonal solver. The width is the maximum
  /// offset (in any dimension) from a cell to another cell whose data needs to
  /// be used by the cell. It is dependent on the implementation.
  constexpr auto width() const {
    return std::size_t{1};
  }

  /// This function solves the \p in_it input data using a constants speed $f=1$
  /// iterator and updates the \p out_it output data. 
  ///
  /// This overload is enabled when the iterator is a GPU iterator.
  ///
  /// \param[in] in_it  The iterable input data to use to evolve.
  /// \param[in] out_it The iteratable output data to update.
  /// \tparam    I      The type of the input iterator.
  template <typename I, typename T, traits::gpu_enable_t<I> = 0>
  fluidity_host_device void solve(I&& it_in, I&& it_out, T dh) const {
    static_assert(is_multidim_iter_v<I>, 
      "Input & output iterators must be a multidimensional iterators!");

    cuda::fast_iterative(std::forward<I>(it_in), std::forward<I>(it_out), dh);
  }


/*
  /// This function solves the \p in_it input data using the \p f_it speed data
  /// iterator and updates the \p out_it output data. This overload is only
  /// enabled when the \p f_it is a multidimensional iterator.
  ///
  /// \param[in] in_it  The iterable input data to use to evolve.
  /// \param[in] out_it The iteratable output data to update.
  /// \param[in] f_it   The iterable speed data.
  /// \param[in] args   Additional arguments for the solver.
  /// \tparam    It     The type of the input iterator.
  /// \tparam    FIt    The type of the speed iterator.
  /// \tparam    Args   The types of any additional arguments.
  template <typename    It  ,
            typename    FIt ,
            typename... Args, multiit_enable_t<FIt> = 0>
  fluidity_host_device void
  solve(It&& it_in, It&& it_out, FIt&& f_it, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>, 
      "Input & output iterators must be a multidimensional iterators!");
    static_assert(is_multidim_iter_v<FIt>, 
      "Speed iterator must be a multidimensional iterator!");

    return impl()->solve_impl(std::forward<It>(it_in)    ,
                              std::forward<It>(it_out)   ,
                              std::forward<FIt>(f_it)    ,
                              std::forward<Args>(args)...);
  }

  /// This function solves the \p in_it input data using the constant \p f speed
  /// data iterator and updates the \p out_it output data. This overload is only
  /// enabled when the \p f is not a multidimensional iterator.
  ///
  /// \note This will fail at compile time if the type of \p f does not match
  ///       the data type accessed by the \p in_it and \p out_it. 
  ///
  /// \param[in] in_it  The iterable input data to use to evolve.
  /// \param[in] out_it The iteratable output data to update.
  /// \param[in] f      The value of the constant speed data.
  /// \param[in] args   Additional arguments for the evolution.
  /// \tparam    It     The type of the input iterator.
  /// \tparam    F      The type of the constant speed data.
  /// \tparam    Args   The types of any additional arguments.
  template <typename    It  ,
            typename    F   ,
            typename... Args, nonmultiit_enable_t<F> = 0>
  fluidity_host_device void
  solve(It&& it_in, It&& it_out, FIt&& f_it, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>, 
      "Input & output iterators must be a multidimensional iterators!");
    static_assert(!is_multidim_iter_v<F>, 
      "Constant speed data cannot be a multidimensional iterator!");
    static_assert(
      std::is_same<std::decay_t<decltype(*it)>, std::decay_t<F>>::value,
      "Constant speed value type must match iterator data type!");

    return impl()->solve_impl(std::forward<It>(it_in)    ,
                              std::forward<It>(it_out)   ,
                              std::forward<F>(f)         ,
                              std::forward<Args>(args)...);
  }

  /// This function solves the \p in_it input data assuming that the speed data
  /// has a constant value of $1$ throught the domain. This overload is defined
  /// due to how common such a case is, for example the levelset equation.
  ///
  /// \param[in] in_it  The iterable input data to use to evolve.
  /// \param[in] out_it The iteratable output data to update.
  /// \param[in] args   Additional arguments for the evolution.
  /// \tparam    It     The type of the input iterator.
  /// \tparam    Args   The types of any additional arguments.
  template <typename It, typename... Args>
  fluidity_host_device void solve(It&& it_in, It&& it_out, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>, 
      "Input & output iterators must be a multidimensional iterators!");

    return impl()->solve_impl(std::forward<It>(it_in)    ,
                              std::forward<It>(it_out)   ,
                              std::forward<Args>(args)...);
  }
*/
};

}}} // namespace fluid::scheme::eikonal

#endif // FLUIDITY_SCHEME_EIKONAL_FAST_ITERATIVE_METHOD_HPP
