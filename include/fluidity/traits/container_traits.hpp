//==--- fluidity/traits/container_traits.hpp --------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  container_traits.hpp
/// \brief This file defines traits related to containers.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_TRAITS_CONTAINER_TRAITS_HPP
#define FLUIDITY_TRAITS_CONTAINER_TRAITS_HPP

#include <type_traits>

namespace fluid  {
namespace traits {
namespace detail {

/// Defines a default class for determining if a class is a container for the
/// false case (that the class is not a container).
/// \tparam T The type to determine if is a container.
/// \tparam _ Default void type.
template<typename T, typename _ = void>
struct IsContainer : std::false_type {};

/// Helper class to determine if a class is a container. This is the default
/// specialization.
/// \tparam Ts The properties which a container must satisfy.
template<typename... Ts>
struct IsContainerHelper {};

/// Specialization of the IsContainer class which specifies the properties which
/// a class T must have to be a container. 
///
/// \note For a class to be a container, it must have an access operator[].
///
/// \tparam T The type to determine if is a container.
template<typename T>
struct IsContainer<
  T,
  std::conditional_t<
    false,
    IsContainerHelper<
      typename T::value_type                   ,
      typename T::size_type                    ,
      typename T::allocator_type               ,
      typename T::iterator                     ,
      typename T::const_iterator               ,
      decltype(std::declval<T>().operator[](0)),
      decltype(std::declval<T>().size())       ,
      decltype(std::declval<T>().begin())      ,
      decltype(std::declval<T>().end())        ,
      decltype(std::declval<T>().cbegin())     ,
      decltype(std::declval<T>().cend())
    >,
    void
  >
> : public std::true_type {};

} // namespace detail

//==--- [Array] ------------------------------------------------------------==//

/// Returns true if the type T is an array, otherwise false.
/// \tparam T The type to determine if is an array.
template <typename T>
static constexpr auto is_array_v = std::is_array<std::decay_t<T>>::value;

//==--- [Container] --------------------------------------------------------==//

/// True when the type T is a container, otherwise false.
/// \tparam T The type to determine if is a container.
template <typename T>
static constexpr auto is_container_v = 
  detail::IsContainer<std::decay_t<T>>::value || is_array_v<std::decay_t<T>>;

/// Wrapper which can be used with SFINAE for enabling functions when a type T
/// is a container.
/// \tparam T The type to base the enable on.
template <typename T>
using container_enable_t = std::enable_if_t<is_container_v<T>, int>;

/// Wrapper which can be used with SFINAE for enabling functions when a type T
/// is not a container.
template <typename T>
using non_container_enable_t = std::enable_if_t<!is_container_v<T>, int>;

namespace detail {

/// Struct to use to overload if a class is a container or not.
/// \tparam IsContainer If the class is a container or not.
template <bool IsContainer>
struct ContainerOverload {
  /// Defines if the class is a container or not.
  static constexpr auto value = IsContainer;
};

} // namespace detail;

/// Defines a class which can be used for overloading when a class is a
/// container.
using container_overload_t = detail::ContainerOverload<true>;

/// Defines a class which can be used for overloading when a class is not a
/// container.
using non_container_overload_t = detail::ContainerOverload<false>;

/// Can be used to select an implementation based on whether or not the class T
/// is a container or not.
/// \tparam T The class to base the overlaoding on.
template <typename T>
using make_container_overload_t = 
  detail::ContainerOverload<is_container_v<std::decay_t<T>>>;

}} // namespace fluid::traits

#endif // FLUIDITY_TRAITS_CONTAINER_TRAITS_HPP
