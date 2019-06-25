//==--- fluidity/utility/type_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  type_traits.hpp
/// \brief This file defines c++17 wrappers so that they can be used with c++14
///        for CUDA code.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_TYPE_TRAITS_HPP
#define FLUIDITY_UTILITY_TYPE_TRAITS_HPP

#include <type_traits>

namespace fluid {

/// Wrapper for std::is_same to allow is_same_v to work with c++14.
/// \tparam A The first type for comparison.
/// \tparam B The second type for comparison.
template <typename A, typename B>
static constexpr bool is_same_v = std::is_same<A, B>::value;

/// Wrapper for std::decay to allow decay_t to work with c++14.
/// \tparam T The type to decay.
template <typename T>
using decay_t = typename std::decay<T>::type;

/// This defines a valid type when T is convertible to U.
/// \tparam T The first data type to check conversion from.
/// \tparam U The second data type to check conversion to.
template <typename T, typename U>
using conv_enable_t = std::enable_if_t<std::is_convertible<T, U>::value, int>;

/// This defines a valid type when T is not convertible to U.
/// \tparam T The first data type to check conversion from.
/// \tparam U The second data type to check conversion to.
template <typename T, typename U>
using conv_disable_t = std::enable_if_t<!std::is_convertible<T, U>::value, int>;

/// This defines a valid type when the type T is integral or when the pack is
/// variadic.
/// \tparam T   The type to check if integral.
/// \tparam Ts  Optional other types if variadic.
template <typename T, typename... Ts>
using var_or_int_enable_t =
  std::enable_if_t<
    std::is_integral<std::decay_t<T>>::value || (sizeof...(Ts) > 0), int>;

/// This defines a valid type when the variadic pack has more than one element.
/// \tparam Ts  Types in the variadic pack.
template <typename... Ts>
using var_enable_t = std::enable_if_t<(sizeof...(Ts) > 1), int>;

/// This defines a valid type when the variadic pack has only a single element,
/// and can be used to disable variadic function overloads.
/// \tparam Ts  Types in the variadic pack.
template <typename... Ts>
using var_disable_t = std::enable_if_t<!(sizeof...(Ts) > 1), int>;

namespace detail {

/// The TypeAt struct defines the type at a given Index in a typelist, with a
/// default type provided if the given index is not in the range of the
/// typelist.
/// \tparam Valid      If the index is valid.
/// \tparam Index      The index of the type in the typelist to get.
/// \tparam Default    The default type to return for an invalid Index.
/// \tparam Typelist   The list of types.
template <bool Valid, std::size_t Index, typename Default, typename Typelist>
struct TypeAt;

/// Specialization of the TypeAt struct for the case that the the index I is out
/// of range for the typelist. In this case the Default type is defined.
/// \tparam I        The index of the element to get.
/// \tparam Default  The default type to use for an invalid index.
/// \tparam Typelist The list of types.
template <std::size_t I, typename Default, typename Typelist>
struct TypeAt<false, I, Default, Typelist>
{
  /// Defines the type to be the default type.
  using type = Default;
};

/// Specialization of the TypeAt struct for the case that the the index I is in
/// range for the tuple, but the tuple contains no elements. In this case the
/// Default type is defined.
/// \tparam I       The index of the element to get.
/// \tparam Default The default type to use for an invalid index.
template <std::size_t I, typename Default>
struct TypeAt<true, I, Default, std::tuple<>>
{
  /// Defines the type to be the default since the there are no elements in the
  /// the type list.
  using type = Default;
};

/// Specialization of the TypeAt struct for the case that the the index I is in
/// range for the tuple.
/// \tparam I       The index of the element to get.
/// \tparam Default The default type to use for an invalid index.
/// \tparam Ts      The types for the tuple.
template <std::size_t I, typename Default, typename... Ts>
struct TypeAt<true, I, Default, std::tuple<Ts...>>
{
  /// Defines the type at position I of the tuple.
  using type = typename std::tuple_element<I, std::tuple<Ts...>>::type;
};

} // namespace detail

/// Returns the type of the element at position I in the variadic parameter
/// pack, if I is within the range of the parameter pack, otherwise the Default
/// type is returned.
/// \tparam I       The index of the type in the pack to get.
/// \tparam Default The type to return when I is out of range.
/// \tparam Ts      The typelist to get a type from.
template <std::size_t I, typename Default, typename... Ts>
using type_at_t = typename
  detail::TypeAt<(I < sizeof...(Ts)), I, Default, std::tuple<Ts...>>::type;

} // namespace fluid

#endif // FLUIDITY_UTILITY_TYPE_TRAITS_HPP
