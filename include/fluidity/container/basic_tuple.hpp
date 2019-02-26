//==--- fluidity/container/basic_tuple.hpp ----------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tuple.hpp
/// \brief This file defines a class for a host-device utility tuple.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_BASIC_TUPLE_HPP
#define FLUIDITY_CONTAINER_BASIC_TUPLE_HPP

#include <fluidity/utility/portability.hpp>
#include <utility>

namespace fluid   {
namespace detail  {

/// The TupleElement stuct stores the index of the element in the tuple.
/// \tparam Index The index of the element.
/// \tparam T     The type of the element.
template <std::size_t Index, typename Opt>
struct TupleElement;

/// This is a helper function which can be decltype'd to get the type of the
/// element at an ndex in a tuple. It should never actually be called, other
/// than in a metaprogramming context.
/// \param[in]  element   The element to get the type of.
/// \tparam     Index     The index of the element in the tuple.
/// \tparam     T         The type to extract from the element.
template <size_t Index, typename T>
fluidity_host_device constexpr inline T
type_extractor(TupleElement<Index, T> element) noexcept
{ 
  return T{};
}

/// Gets the element \p e at position I in a tuple.
/// \param[in] e  The element to get.
/// \tparam    I  The index of the element to get.
/// \tparam    T  The type of the element to get.
template <std::size_t I, typename T>
fluidity_host_device constexpr inline auto&&
get(TupleElement<I, T>&& e)
{
  return e.value;
}

/// Gets the element \p e at position I in a tuple.
/// \param[in] e  The element to get.
/// \tparam    I  The index of the element to get.
/// \tparam    T  The type of the element to get.
template <std::size_t I, typename T>
fluidity_host_device constexpr inline const auto&
get(const TupleElement<I, T>& e)
{
  return e.value;
}

/// Gets the element \p e at position I in a tuple.
/// \param[in] e  The element to get.
/// \tparam    I  The index of the element to get.
/// \tparam    T  The type of the element to get.
template <std::size_t I, typename T>
fluidity_host_devcice constexpr inline auto&
get(TupleElement<I, T>& e)
{
  return e.value;
}

/// The TupleElement stuct stores the index of the option in the tuple.
/// \tparam Index The index of the element.
/// \tparam T     The type of the element.
template <std::size_t Index, typename T>
struct TupleElement
{
  /// Defines the type of the option.
  using type_t = T;

  /// Defines the index of this option.
  static constexpr auto index = Index;

  /// Default constructor for the element.
  fluidity_host_device constexpr TupleElement() = default;

  /// Constructor to move an element into this one.
  template <typename T>
  fluidity_host_device constexpr TupleElement(T&& v)
  : value(std::forward<T>(v)) {}

  type_t value = type_t{0}; //!< The element for the tuple
};

/// The TupleStorage struct implements a container to store different types.
/// \tparam Indices  The indices of the elements.
/// \tparam Elements The types of the elements.
template <typename Indices, typename... Elements>
struct TupleStorage;

/// The TupleStorage struct implements a container for different types.
/// \tparam I   The indices of the elements.
/// \tparam Ts  The type of the elements.
template <std::size_t... I, typename... Ts>
struct TupleStorage<std::index_sequence<I...>, Ts...> : TupleElement<I, Ts>... {
  /// Defines the number of options in the tuple.
  static constexpr auto elements = sizeof...(Ts);

  /// Default constructor -- allows the creation of an empty tuple.
  constexpr TupleStorage() = default;

  /// Constructor which takes a list of elements to use to create the tuple.
  template <typename... Es>
  fluidity_host_device constexpr TupleStorage(Es&&... es)
  : TupleElement<I, Ts>(std::forward<Es>(es))... {}

  /// Constructor which takes a list of elements to use to create the tuple.
  template <typename... Es>
  fluidity_host_device constexpr TupleStorage(const Es&... es)
  : TupleElement<I, Ts>(std::forward<Ts>(es))... {}
};

} // namespace detail

/// The BasicTuple struct implements a container for storing a list of different
/// types. 
/// \tparam Es The types to store.
template <typename... Es>
struct BasicTuple final :
detail::TupleStorage<std::make_index_sequence<sizeof...(Es)>, Es...> {
  /// Defines the type of the index sequence.
  using indices_t = std::make_index_sequence<sizeof...(Es)>;
  /// Defines the type of the base imlpementation.
  using storage_t = detail::TupleStorage<indices_t, Es...>;

  /// Defines the number of elements in the option tuple.
  static constexpr auto elements = sizeof...(Es);

  /// Default constructor which allows the creation of a deafault tuple.
  constexpr BasicTuple() = default;

  /// Constructor to create the tuple from a list of types.
  template <typename... Ts>
  constexpr BasicTuple(Ts&&... ts) : storage_t{std::forward<Ts>(ts)...} {}
};

} // namespace fluid

#endif // FLUIDITY_CONTAINER_BASIC_TUPLE_HPP