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

#include "limiting_impl.hpp"

namespace fluid  {
namespace limit  {

/// Computes the backward difference of the state pointed to by the iterator and
/// state behind it, in the given dimension. The state variables are  
/// differenced in the \p requested_form. If the form of the state is different
/// from the \p requested_form, then they are trasformed before the differencing
/// and are returned in the \p requested_form.
/// \param[in] state_it       An iterator to the state.
/// \param[in] mat            The material for the system.
/// \param[in] requested_form The form of the state for differencing.
/// \tparam    Form           The form of the variables to limit on.
/// \tparam    Iterator       The type of the state iterator.
/// \tparam    Material       The type of the material
/// \tparam    Value          The value which defines the dimension. 
template < typename    Form
         , typename    Iterator
         , typename    Material
         , std::size_t Value>
fluidity_host_device constexpr auto
backward_diff(Iterator&& state_it, Material&& mat, Dimension<Value>)
{
  return detail::backward_diff<Form>(std::forward<Iterator>(state_it),
                                     std::forward<Material>(mat)     ,
                                     Dimension<Value>{}              );
}

/// Computes the forward difference between the state ahead of the state
/// pointed to by the iterator and the state pointed to by the iterator, in the
/// given dimension. The state variables are differenced in the \p 
/// requested_form. If the form of the state is different from the \p
/// requested_form, then they are trasformed before the differencing
/// and are returned in the \p requested_form.
/// \param[in] state_it       An iterator to the state.
/// \param[in] mat            The material for the system.
/// \param[in] requested_form The form of the state for differencing.
/// \tparam    Form           The form of the variables to limit on.
/// \tparam    Iterator       The type of the state iterator.
/// \tparam    Material       The type of the material
/// \tparam    Value          The value which defines the dimension. 
template < typename    Form
         , typename    Iterator
         , typename    Material
         , std::size_t Value>
fluidity_host_device constexpr auto
forward_diff(Iterator&& state_it, Material&& mat, Dimension<Value>)
{
  return detail::forward_diff<Form>(std::forward<Iterator>(state_it),
                                    std::forward<Material>(mat)     ,
                                    Dimension<Value>{}              );
}

/// Computes the central difference between the state ahead of the state
/// pointed to by the iterator and the state behind the state pointed to by the
/// iterator, in the given dimension. The state variables are differenced in
/// the \p requested_form. If the form of the state is different from the \p
/// requested_form, then they are trasformed before the differencing
/// and are returned in the \p requested_form.
/// \param[in] state_it       An iterator to the state.
/// \param[in] mat            The material for the system.
/// \param[in] requested_form The form of the state for differencing.
/// \tparam    Iterator       The type of the state iterator.
/// \tparam    Material       The type of the material
/// \tparam    Form           The form of the variables to limit on.
/// \tparam    Value          The value which defines the dimension. 
template < typename    Form
         , typename    Iterator
         , typename    Material
         , std::size_t Value>
fluidity_host_device constexpr auto
central_diff(Iterator&& state_it, Material&& mat, Dimension<Value>)
{
  return detail::central_diff<Form>(std::forward<Iterator>(state_it),
                                    std::forward<Material>(mat)     ,
                                    Dimension<Value>{}              );
}

}} // namespace fluid::limit

#endif // FLUIDITY_LIMITING_LIMITING_HPP