//==--- fluidity/limiting/limiting_impl.hpp ---------------- -*- C++ -*- ---==//
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

#ifndef FLUIDITY_LIMITING_LIMITING_IMPL_HPP
#define FLUIDITY_LIMITING_LIMITING_IMPL_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace limit  {
namespace detail {

/// Used to enable overloads when the state type held by the iterator has the
/// same form as that requested by the calling function.
/// \tparam Iterator      The type of the state iterator.
/// \tparam RequestedForm The requested form for the state.
template <typename Iterator, typename RequestedForm>
using enable_same_t = 
  std::enable_if_t<iter_value_t<Iterator>::format == RequestedForm::form, int>;

/// Used to enable overloads when the state type held by the iterator does not
/// have the same form as that requested by the calling function, and the
/// requested form is primitive.
/// \tparam Iterator      The type of the state iterator.
/// \tparam RequestedForm The requested form for the state.
template <typename Iterator, typename RequestedForm>
using enable_different_primitive_req_t =
  std::enable_if_t<
    iter_value_t<Iterator>::format != RequestedForm::form &&
    RequestedForm::form            == state::FormType::primitive, int
  >;

/// Used to enable overloads when the state type held by the iterator does not
/// have the same form as that requested by the calling function, and the
/// requested form is conservative.
/// \tparam Iterator      The type of the state iterator.
/// \tparam RequestedForm The requested form for the state.
template <typename Iterator, typename RequestedForm>
using enable_different_conservative_req_t =
  std::enable_if_t<
    iter_value_t<Iterator>::format != RequestedForm::form &&
    RequestedForm::form            == state::FormType::conservative, int
  >;

//==--- Backward difference ------------------------------------------------==//

/// Implementation of the backward difference between the state the \p state_it
/// points to and the state behind it. This overload is for when the type of the
/// iterated state is conservative but limiting must be done on the primitive
/// form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_primitive_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
backward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it->primitive(mat) - state_it.offset(-1, dim)->primitive(mat);
}

/// Implementation of the backward difference between the state the \p state_it
/// points to and the state behind it. This overload is for when the type of the
/// iterated state is primitive but limiting must be done on the conservative
/// form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_conservative_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
backward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it->conservative(mat)
       - state_it.offset(-1, dim)->conservative(mat);
}

/// Implementation of the backward difference between the state the \p state_it
/// points to and the state behind it. This overload is for when the form of the
/// iterated state is the same as the requested form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_same_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
backward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.backward_diff(dim);
}

//==--- Forward difference -------------------------------------------------==//

/// Implementation of the forward difference between the state the \p state_it
/// points to and the state ahead of it. This overload is for when the type of
/// the iterated state is conservative but limiting must be done on the 
/// primitive form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_primitive_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
forward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.offset(1, dim)->primitive(mat) - state_it->primitive(mat);
}

/// Implementation of the forward difference between the state the \p state_it
/// points to and the state ahead of it. This overload is for when the type of 
/// the iterated state is primitive but limiting must be done on the 
/// conservative form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_conservative_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
forward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.offset(1, dim)->conservative(mat)
       - state_it->conservative(mat);
}

/// Implementation of the forward difference between the state the \p state_it
/// points to and the state ahead of it. This overload is for when the form of
/// the iterated state is the same as the requested form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_same_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
forward_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.forward_diff(dim);
}

//==--- Central difference -------------------------------------------------==//

/// Implementation of the central difference between the state ahead of the
/// state which the \p state_it points to and the state behind it. This
/// overload is for when the type of the iterated state is conservative but
/// limiting must be done on the primitive form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_primitive_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
central_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.offset( 1, dim)->primitive(mat) - 
         state_it.offset(-1, dim)->primitive(mat);
}

/// Implementation of the central difference between the state ahead of the
/// state which the \p state_it points to and the state behind it. This
/// overload is for when the type of the iterated state is primitive but
/// limiting must be done on the conservative form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Value    The value which defines the dimension to limit in.
template < typename    Form
         , typename    Iterator
         , typename    Material
         , typename    Dim
         , enable_different_conservative_req_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
central_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.offset( 1, dim)->conservative(mat)
       - state_it.offset(-1, dim)->conservative(mat);
}

/// Implementation of the central difference between the state ahead of the
/// state which is the \p state_it points to and the state behind it. This
/// overload is for when the form of the iterated state is the same as the
/// requested form.
/// \param[in] state_it The state iterator.
/// \param[in] mat      The material for the system.
/// \tparam    Iterator The type of the state iterator.
/// \tparam    Material The type of the material.
/// \tparam    Form     The requested form for limiting.
/// \tparam    Value    The value which defines the dimension to limit in.
  template < typename    Form
           , typename    Iterator
           , typename    Material
           , typename    Dim
           , enable_same_t<Iterator, Form> = 0>
fluidity_host_device constexpr auto
central_diff(Iterator&& state_it, Material&& mat, Dim dim)
{
  return state_it.central_diff(dim);
}

}}} // namespace fluid::limit::detail

#endif // FLUIDITY_LIMITING_LIMITING_IMPL_HPP