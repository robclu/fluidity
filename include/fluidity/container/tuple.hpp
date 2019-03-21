//==--- fluidity/container/tuple.hpp ----------------------- -*- C++ -*- ---==//
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
/// \brief This file defines a class for a host-device tuple.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_TUPLE_HPP
#define FLUIDITY_CONTAINER_TUPLE_HPP

#include "basic_tuple.hpp"

namespace fluid {

/// Forward declaration of the Tuple class.
/// \tparam Ts The types for the tuple.
template <typename... Ts>
struct Tuple;

namespace detail {

/// The IsTuple struct determines if a type is a Tuple. This is the default case
/// which returns false.
/// \tparam T   The type to check against tuple.
template <typename T>
struct IsTuple {
  /// Returns that the type T is not a tuple.
  static constexpr bool value = false;
};

/// Specialization of the IsTuple struct to return that a type is a Tuple.
/// \tparam Ts  The types the tuple stores.
template <typename... Ts>
struct IsTuple<Tuple<Ts...>> {
  /// Returns that the type is a tuple.
  static constexpr bool value = true;
};

/// Struct to define the size of a tuple. This non-specialization is the general
/// case where T is not a tuple and therefore has no size.
/// \tparam T The tuple to determine the size of.
template<typename T>
struct TupleSize {
  /// Returns that the tuple has no size (non-tuple specialization).
  static constexpr auto value = std::size_t{0};
};

/// Specialization of the TupleSize struct for the case that the type is a
/// Tuple.
/// \tparam Ts The types for the tuple.
template <typename... Ts>
struct TupleSize<Tuple<Ts...>> {
  /// Returns the size of the tuple.
  static constexpr auto value = std::size_t{sizeof...(Ts)};
};

} // namespace detail

//==--- Traits -------------------------------------------------------------==//

/// Returns the size of the tuple.
template <typename T>
static constexpr auto tuple_size_v = detail::TupleSize<std::decay_t<T>>::value;

/// Returns true if a decayed T is a tuple, and false otherwise.
template <typename T>
static constexpr bool is_tuple_v = detail::IsTuple<std::decay_t<T>>::value;

/// Defines a valid type if the Es are a tuple.
/// \tparam Es the type(s) to check if are a tuple.
template <typename... Es>
using tuple_enable_t = std::enable_if_t<is_tuple_v<Es...>>;
  
/// Defines a valid type if the Es aren't a tuple.
template <typename... Es>
using nontuple_enable_t = std::enable_if_t<!is_tuple_v<Es...>>;

//==--- Tuple Implementation -----------------------------------------------==//

/// Specialization for an empty Tuple.
template <>
struct Tuple<> {
  /// Alias for the storage type.
  using storage_t = BasicTuple<>;
  /// Defines the size of the Tuple.
  static constexpr size_t elements = storage_t::elements;

  /// Intializes the Tuple with no elements.
  fluidity_host_device constexpr Tuple() {}

  /// Retutns the size (0) of the tuple.
  constexpr size_t size() const { return 0; }
};

/// Specialization for a non-empty Tuple.
/// \tparam Ts The types of the Tuple elements.
template <typename... Ts> 
struct Tuple {
 private:

 public:
  /// Alias for the storage type.
  using storage_t = BasicTuple<Ts...>;

  /// Defines the number of elements in the tuple.
  static constexpr size_t elements = storage_t::elements;

  /// Default constructor -- creates default initialized elements.
  fluidity_host_device constexpr Tuple()
  : _storage{std::decay_t<Ts>()...} {}

  /// Intializes the Tuple with a variadic list of lvalue elements.
  /// \param[in] elements The elements to store in the Tuple.
  fluidity_host_device constexpr Tuple(const Ts&... es)
  : _storage{es...} {}

  /// Initializes the Tuple with a variadic list of forwarding reference
  /// elements. This overload is only selcted if the Es are not a tuple, but are
  /// the types held by the tuple.
  /// \param[in] elements  The elements to store in the Tuple.
  /// \tparam    Es        The types of the elements.
  template <typename... Es, nontuple_enable_t<Es...> = 0>
  fluidity_host_device constexpr Tuple(Es&&... es)
  : _storage{std::forward<Es>(es)...} {}

  /// Copy and move constructs the Tuple. This overload is only selcted if the
  /// type T matches this Tuple's type, i.e for copy and move construction.
  ///  
  /// \param[in] other  The other tuple to copy or move.
  /// \tparam    T      The type of the other tuple.
  template <typename T, tuple_enable_t<T> = 0>
  fluidity_host_device constexpr explicit Tuple(T&& other)
  : Tuple{std::make_index_sequence<elements>{}, std::forward<T>(other)} {}

  //==--- Methods ----------------------------------------------------------==//
  
  /// Returns the number of elements in the tuple.
  fluidity_host_device constexpr std::size_t size() const
  {
    return elements;
  }

  /// Returns the underlying storage container, which holds the elements.
  fluidity_host_device storage_t& data()
  { 
    return _storage; 
  }

  /// Returns a constant reference to the underlying storage container,
  /// which holds the elements.
  fluidity_host_device const storage_t& data() const
  { 
    return _storage;
  }

  /// Returns the underlying storage container, which holds the elements.
  fluidity_host_device volatile storage_t& data() volatile
  { 
    return _storage; 
  }

  /// Returns a constant reference to the underlying storage container,
  /// which holds the elements.
  fluidity_host_device const volatile storage_t& data() const volatile
  { 
    return _storage;
  }

 private:
  storage_t _storage; //!< Storage of the Tuple elements.


  /// This overload of the constructor is called by the copy and move
  /// constructores to get the elements of the \p other Tuple and copy or move
  /// them into this Tuple.
  /// 
  /// \param[in] extractor Used to extract the elements out of \p other.
  /// \param[in] other     The other tuple to copy or move.
  /// \tparam    T         The type of the other tuple.
  template <std::size_t... I, typename T>
  fluidity_host_device constexpr explicit
  Tuple(std::index_sequence<I...> extractor, T&& other)
  : _storage{detail::get<I>(std::forward<storage_t>(other.data()))...} {}
};

namespace detail {

/// The TupleElement class get the type of the element at index Idx in a tuple.
/// \tparam   Idx  The index of the element to get type type of.
/// \tparam   T   The type of the tuple.
template <std::size_t Idx, typename T>
struct TupleElementType {
  /// Returns the type of the element at index Idx.
  using type = decltype(
    detail::type_extractor<Idx>(std::move(std::declval<T>().data())));
};

/// Returns the type of a tuple element for a tuple with no elements,
/// \tparam  Idx The index of the element to get the type of.
template <std::size_t Idx>
struct TupleElementType<Idx, Tuple<>> {
  using type = void;
};

} // namespace detail

/// Defines the type of the Ith element in a tuple.
/// \tparam I The index of the element to get the type of.
/// \tparam T The type of the tuple to get the element type from.
template <std::size_t I, typename T>
using tuple_element_t = typename detail::TupleElementType<I, T>::type;

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a const lvalue reference.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    I     The index of the element to get from the Tuple.
/// \tparam    Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
fluidity_host_device constexpr inline const auto& get(const Tuple<Ts...>& tuple)
{
  return detail::get<I>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a reference type.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    I     The index of the element to get from the Tuple.
/// \tparam    Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
fluidity_host_device constexpr inline auto& get(Tuple<Ts...>& tuple)
{
  return detail::get<I>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a forwarding reference type.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    I     The index of the element to get from the Tuple.
/// \tparam    Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
fluidity_host_device constexpr inline auto&& get(Tuple<Ts...>&& tuple)
{
  return std::move(detail::get<I>(std::move(tuple.data())));
}

//==--- Functions ----------------------------------------------------------==//

/// This makes a tuple, and is the interface through which tuples should be
/// created in almost all cases. Example usage is:
///
/// ~~~cpp
/// auto tuple = make_tuple(4, 3.5, "some value");  
/// ~~~
/// 
/// This imlementation decays the types, so it will not create refrence types,
/// i.e:
/// 
/// ~~~cpp
/// int x = 4, y = 5;
/// auto tup = make_tuple(x, y);
/// ~~~
/// 
/// will copy ``x`` and ``y`` and not created a tuple of references to the
/// variables. However, that kind of desired behaviour is one of the instances
/// when explicit creation of the tuple is required:
/// 
/// ~~~cpp
/// int x = 0, y = 0.0f;
/// Tuple<int&, float&> tuple = make_tuple(x, y);
/// 
/// // Can modify x and y though tuple:
/// get<0>(tuple) = 4;
/// tuple.at<1>() = 3.5f;
/// 
/// // Can modify tuple values though x and y:
/// x = 0; y = 0.0f;
/// ~~~
/// 
/// \param[in]  values  The values to store in the tuple.
/// \tparam     Ts      The types of the values, and which will define the type
///                     of the Tuple.
template <typename... Ts>
fluidity_host_device constexpr inline auto make_tuple(Ts&&... values)
{
  return Tuple<Ts...>(std::forward<Ts>(values)...);
}

namespace detail {

/// Implementation of the unpack function which expands the tuple into the
/// callable function \p fn. This is different to $for_each$ in that $for_each$
/// will apply a function to each element, where this expands the entire tuple
/// contents into the function.
/// \param[in] fn The function to expand the tuple into.
/// \param[in] t  The tuple to unpack.
/// \tparam    F  The type of the function.
/// \tparam    T  The type of the tuple.
/// \tparam    I  The indices of the tuple elements.
template<typename F, typename T, size_t... I>
fluidity_host_device auto unpack_impl(F&& fn, T&& t, std::index_sequence<I...>)
{
  return fn(get<I>(std::forward<T>(t))...);
}

} // namespace detail

/// Implementation of the unpack function which expands the tuple into the
/// callable function \p fn. This is different to $for_each$ in that $for_each$
/// will apply a function to each element, where this expands the entire tuple
/// contents into the function. For example, this may be used as follows:
/// \code{cpp}
/// auto tuple  = make_tuple(...); // Some example tuple
/// auto result = 
///   unpack(tuple, [] (auto&&... args)
///   {
///     // Do something with tuple args...
///   });
/// \endcode
///
/// \param[in] fn    The function to expand the tuple into.
/// \param[in] t     The tuple to unpack.
/// \tparam    T     The tuple to
/// \tparam    F     The type of the function.
/// \tparam    Args  The indices of the tuple elements.
template<typename T, typename F, typename... Args>
fluidity_host_device auto unpack(T&& t, F&& fn)
{
  return apply_tuple_impl(std::forward<F>(fn),
                          std::forward<T>(t),
                          std::make_index_sequence<tuple_size_v<T>>());
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_TUPLE_HPP