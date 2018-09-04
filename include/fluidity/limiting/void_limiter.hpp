//==--- fluidity/limiting/void_limiter.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  void_limiter.hpp
/// \brief This file defines an implementation of a limiter which does nothing.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_VOID_LIMITER_HPP
#define FLUIDITY_LIMITING_VOID_LIMITER_HPP

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace limit {

/// The Void limiter class defines a functor which just returns the state.
/// \tparam Form The form of the limiting.
template <typename Form>
struct Void {
  /// Defines the type of this class.
  using self_t = Void;
  /// Defines the form of the variables to limit on.
  using form_t = Form;

  /// Defines the number of elements required for limiting.
  static constexpr std::size_t width = 1;

  /// Implementation of the linear limiting functionality.
  /// \param[in]  state  The state iterator to limit.
  /// \tparam     IT     The type of the state iterator.
  /// \tparam     Dim    The type of the dimension. 
  template <typename IT, typename Dim>
  fluidity_host_device constexpr auto operator()(IT&& state, Dim) const
  {
    return *state;
  }
};

}} // namespace fluid::limit


#endif // FLUIDITY_LIMITING_VOID_LIMITER_HPP
