//==--- fluidity/scheme/interfaces/eikonal.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eikonal.hpp
/// \brief This file defines an interface for a class which computes the
///        solution to the eikonal equation.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_EIKONAL_HPP
#define FLUIDITY_SCHEME_INTERFACES_EIKONAL_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

/// The Eikonal class provides an interface to which numerical methods for
/// solving the Eikonal equation must conform. The Eikonal equation is defined
/// as:
///
/// \begin{equation}
///   |\nabla \phi(x)| = \frac{1}{f(x)}
/// \end{equation}
///
/// where $f(x)$ is the speed function (with positive values) and is the speed
/// at $x$.
/// 
/// The specific implementation of the Eikonal solver scheme is defined by
/// template type. 
///
/// \tparam EikonalImpl The implementation of the Eikonal interface.
template <typename EikonalImpl>
class Eikonal {
  /// Defines the type of the Eikonal implementation.
  using impl_t = EikonalImpl;

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

 public:
  /// Returns the width required by the Eikonal solver. The width is the maximum
  /// offset (in any dimension) from a cell to another cell whose data needs to
  /// be used by the cell. It is dependent on the implementation.
  constexpr auto width() const
  {
    return impl()->width();
  }

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
  solve(It&& it_in, It&& it_out, F& f, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>, 
      "Input & output iterators must be a multidimensional iterators!");
    static_assert(!is_multidim_iter_v<F>, 
      "Constant speed data cannot be a multidimensional iterator!");

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
};

//==--- [Traits] -----------------------------------------------------------==//

/// Returns true if the type T conforms to the Eikonal interface.
/// \tparam T The type to check for conformity to the Eikonal inteface.
template <typename T>
static constexpr auto is_eikonal_v = 
  std::is_base_of<Eikonal<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_INTERFACES_EIKONAL_HPP
