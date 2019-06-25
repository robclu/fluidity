//==--- fluidity/limiting/limiter.hpp- -----------------------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limiter.hpp
/// \brief This file defines the interface to which all limiters must conform,
///        and also provides a default implementation which derived limters can
///        invoke if their implementation is not specific.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_LIMITER_HPP
#define FLUIDITY_LIMITING_LIMITER_HPP

#include "limiting.hpp"
#include "limiter_traits.hpp"
#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/container/array.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace limit {

/// The Limiter class defines the interface to which all limiters
/// must conform. The implementation is provided by the template type.
/// \tparam LimiterImpl The type of the limiter implementation.
template <typename LimiterImpl>
class Limiter
{
  /// Defines the type of the reconstructor implementation.
  using impl_t   = LimiterImpl;
  /// Defines the type of the traits for the limiter.
  using traits_t = LimiterTraits<impl_t>;
  /// Defines the form of the limiting (which variables are limited on which
  /// variables.)
  using form_t   = typename traits_t::form_t;

  /// Constructor, which is made private so that there is no bug if some
  /// limiter inherits from another one, i.e, given Impl1 and Impl2,
  /// then:
  /// \begin{code}
  ///   class Impl1 : Limiter<Impl1> { ... };
  ///
  ///   class Impl2 : Limter<Impl1> { ... }; // Error without fix.
  /// \end{code}
  //fluidity_host_device constexpr Limiter() {};

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
  /// Defines the number of elements required for limiting.
  static constexpr auto width = traits_t::width;

  /// Default implementation of the limiting, which requires that the
  /// implementation type has a `limit_single(left, right)` implementation.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  material  The material for the system.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Material  The type of the material.
  /// \tparam     Value     The value which defines the dimension to limit in.
  template <typename Iterator, typename Material, typename Dim>
  fluidity_host_device constexpr auto
  limit_generic(Iterator&& state_it, Material&& mat, Dim dim) const
  {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = typename state_t::value_t;
    
    const auto fwrd_diff = forward_diff<form_t>(state_it, mat, dim);
    const auto back_diff = backward_diff<form_t>(state_it, mat, dim);
    auto limited = value_t{0.5}
                 * (state_it.backward_diff(dim)
                 +  state_it.forward_diff(dim));
                
    unrolled_for<state_t::elements>([&] (auto i)
    {
      limited[i] *= impl()->limit_single(back_diff[i], fwrd_diff[i]);
    });
    return limited;
  }

  /// Defines an instance of the form of the variabl
  /// Limits the data pointed to by the \p state_it in the given dimension.
  /// \param[in]  state_it  The state iterator to limit.
  /// \param[in]  material  The material for the system.
  /// \tparam     Iterator  The type of the state iterator.
  /// \tparam     Material  The type of the material.
  /// \tparam     Value     The value which defines the dimension to limit in.
  template <typename Iterator, typename Material, typename Dim>
  fluidity_host_device constexpr auto
  limit(Iterator&& state_it, Material&& mat, Dim dim) const
  {
    return impl()->limit_impl(std::forward<Iterator>(state_it),
                              std::forward<Material>(mat)     ,
                              dim                             );
  }
};

}} // namespace fluid::limit

#endif // FLUIDITY_LIMITING_LIMITER_HPP