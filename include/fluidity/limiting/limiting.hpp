//==--- fluidity/limiting/limiting.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limiting.hpp
/// \brief This file includes general limting functionality which can be used
///        in the implementation of different limiters.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_LIMITING_HPP
#define FLUIDITY_LIMITING_LIMITING_HPP

#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace limit  {

/// Used to enable overloads when the state type held by the iterator has the
/// same form as that requested by the calling function.
/// \tparam Iterator      The type of the state iterator.
/// \tparam RequestedForm The requested form for the state.
template <typename Iterator, state::FormType RequestedForm>
using enable_same_t = 
  std::enable_if_t<
    is_same_v<iter_value_t<Iterator>::format, RequestedForm>, int
  >;

/// Used to enable overloads when the state type held by the iterator does not
/// have the same form as that requested by the calling function.
/// \tparam Iterator      The type of the state iterator.
/// \tparam RequestedForm The requested form for the state.
template <typename Iterator, state::FormType RequestedForm>
using enable_different_t =
  std::enable_if_t<
    !is_same_v<iter_value_t<Iterator>::format, RequestedForm>, int
  >;

/// Computes the backward difference of the state pointed to by the iterator and
/// state ahead of it, in the given dimension. The state variables are
/// differenced in the \p requested_form. If the form of the state is different
/// from the \p requested_form, then they are trasformed before the differencing
/// and then transformed back afterwards.
/// \param[in] state_it   An iterator to the state.
/// \param[in] requested_form The form of the state for differencing.
/// \tparam    Iterator       The type of the state iterator.
/// \tparam    Value          The value which defines the dimension. 
template <typename Iterator, std::size_t Value>
fluidity_host_device constexpr auto
backward_diff(Iterator&&      state_it      ,
              state::FormType requested_form,
              Dimension<Value> /*dim*/      )
{
  // Forward to the appropriate overload ...
}

}} // namespace fluid::limit

#endif // FLUIDITY_LIMITING_LIMITING_HPP