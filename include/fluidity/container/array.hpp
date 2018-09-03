//==--- fluidity/container/array.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array.hpp
/// \brief This file defines the implementation of an array class.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_ARRAY_HPP
#define FLUIDITY_CONTAINER_ARRAY_HPP

#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/iterator/strided_iterator.hpp>

namespace fluid {

/// The Array class defined a 1 dimensional, contiguous container with a size
/// known at compile time. 
/// \tparam T         The type of the data to store in the tensor.
/// \tparam Elements  The number of elements in the array.
template <typename T, std::size_t Elements>
class Array {
 private:
  /// Defines the cutoff size for small versions of the array.
  static constexpr auto small_size = std::size_t{8};

  /// Defines a valid type if the variadic pack has more than one element.
  /// \tparam Values The values in the list.
  template <typename... Values>
  using list_enable_t = std::enable_if_t<(sizeof...(Values) > 1), int>;
  /// Defines a valid type if the size is less than the small size.
  /// \tparam S The size to base the enable on.
  template <std::size_t S>
  using small_enable_t = std::enable_if_t<(S <= small_size), int>;

  /// Defines a valid type if the size is greater than the small size.
  /// \tparam S The size to base the enable on.
  template <std::size_t S>
  using small_disable_t = std::enable_if_t<(S > small_size), int>;

 public:
  /// Defines the type of the tensor.
  using self_t            = Array;
  /// Defines the type of the elements in the tensor.
  using value_t           = std::decay_t<T>;
  /// Defines the type of the pointer to the data type.
  using pointer_t         = value_t*;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&; 
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&;
  /// Defines the type of a non const iterator.
  using iterator_t        = StridedIterator<value_t, false>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = StridedIterator<value_t, true>;

  /// Defines the number of elements in the array.
  static constexpr auto elements   = Elements;

  /// The default constructor uses the default initialization.
  constexpr Array() = default;

  /// Initializes each of the elements in the array to have the value \p value.
  /// \param[in] value The value to set the array elements to.
  fluidity_host_device constexpr Array(value_t value)
  {
    initialize<elements>(value);
  }

  /// Initializes the elements in the array to have the values \p values.
  /// \param[in] values The values to set the array elements to.
  /// \tparam    Values The types of the values.
  template <typename... Values, list_enable_t<Values...> = 0>
  fluidity_host_device constexpr Array(Values&&... values)
  : _data{static_cast<value_t>(values)...} {}

  /// Copies the contents of the \p other array into a new array.
  /// \param[in] other The other array to copy from.
  constexpr Array(const self_t& other) = default;

  /// Moves the \p other array into this one.
  /// \param[in] other The other array to move from.
  constexpr Array(self_t&& other) = default;

  /// Copies the contents of the \p other array into a new array.
  /// \param[in] other The other array to copy from.
  constexpr self_t& operator=(const self_t&) = default;

  /// Moves the \p other array into this one.
  /// \param[in] other The other array to move from.
  constexpr self_t& operator=(self_t&&) = default;

  /// Overload of access operator to access an element. __Note:__
  /// this does not check that the value of \p i is in range.
  /// \param[in] i The index of the element in the vetor to return.
  fluidity_host_device constexpr reference_t operator[](size_t i)
  {
    return _data[i];
  }

  /// Overload of access operator to access an element in the array. __Note:__
  /// this does not check that the value of \p i is in range.
  /// \param[in] i The index of the element in the vetor to return.
  fluidity_host_device constexpr const_reference_t operator[](size_t i) const
  {
    return _data[i];
  }

  /// Overload of operator- to subtract each element of a container from each
  /// element in the array, returning a new array.
  /// \param[in] container The container to elementwise subtract from the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t operator-(Container&& container) const
  {
    auto result = *this;
    unrolled_for_bounded<max_unroll_depth>([&] (auto i)
    {
      result[i] -= container[i];
    });
    return result;
  }

  /// Overload of operator+ to add each element of a container to each element
  /// in the array, returning a new array.
  /// \param[in] container The container to elementwise add with the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t operator+(Container&& container) const
  {
    auto result = *this;
    unrolled_for_bounded<max_unroll_depth>([&] (auto i)
    {
      result[i] += container[i];
    });
    return result;
  }

  /// Overload of operator-= to subtract each element of a container from each
  /// element in the array, returning the modified array.
  /// \param[in] container The container to elementwise subtract from the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t& operator-=(Container&& container)
  {
    unrolled_for_bounded<max_unroll_depth>([&] (auto i)
    {
      this->operator[](i) -= container[i];
    });
    return *this;
  }

  /// Overload of operator+= to add each element of a container to each element
  /// in the array, returning the modified array.
  /// \param[in] container The container to elementwise add with the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t& operator+=(Container&& container)
  {
    unrolled_for_bounded<max_unroll_depth>([&] (auto i)
    {
      this->operator[](i) += container[i];
    });
    return *this;
  }

  /// Returns an iterator to the first element in the tensor.
  fluidity_host_device constexpr iterator_t begin()
  {
    return iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  fluidity_host_device constexpr iterator_t end()
  {
    return iterator_t{this->_data + Elements};
  }

  /// Returns a constant iterator to the first element in the array.
  fluidity_host_device constexpr const_iterator_t begin() const
  {
    return const_iterator_t{this->_data};
  }

  /// Returns a constant iterator to the last element in the array.
  fluidity_host_device constexpr const_iterator_t end() const
  {
    return const_iterator_t{this->_data + Elements};
  }

  /// Returns the number of elements in the array.
  fluidity_host_device constexpr std::size_t size() const
  {
    return Elements;
  }

 private:
  value_t _data[Elements] = {0};  //!< Data for the array.

  /// Fill implementation for small arrays, where all elements of the array are
  /// filled with the value \p value in an unrolled manner.
  /// \param[in] value The value to set the array elements to.
  /// \tparam    E     The number of elements in the array.
  template <std::size_t E, small_enable_t<E> = 0>
  fluidity_host_device constexpr void initialize(T value)
  {
    unrolled_for<E>([&, this] /*fluidity_host_device*/ (auto i)
    {
      this->_data[i] = value;
    });
  }

  /// Fill implementation for large arrays, where all elements of the array are
  /// filled with the value \p value.
  /// \param[in] value The value to set the array elements to.
  /// \tparam    E     The number of elements in the array.
  template <std::size_t E, small_disable_t<E> = 0>
  fluidity_host_device constexpr void initialize(T value)
  {
    for (auto i : range(E))
    {
      this->_data[i] = value;
    }
  }
};

//==--- Array Implementation -----------------------------------------------==//

//===== Operator --------------------------------------------------------=====//

/// Overload of multiplication operator to perform elementwise multiplication of
/// a scalar constant to an array.
/// \param[in] scalar The scalar to multiply to each element of the array.
/// \param[in] a      The array to multiply with the scalar.
/// \tparam    T      The type of the scalar data.
/// \tparam    U      The type of the data in the array.
/// \tparam    S      The size of the array.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator*(T scalar, const Array<U, S>& a)
{
  auto result = a;
  unrolled_for_bounded<max_unroll_depth>([&] (auto i)
  {
    result[i] *= scalar;
  });
  return result;
} 

/// Overload of devision operator to perform elementwise multiplication of
/// a scalar constant to an array.
/// \param[in] scalar The scalar to multiply to each element of the array.
/// \param[in] a      The array to multiply with the scalar.
/// \tparam    T      The type of the scalar data.
/// \tparam    U      The type of the data in the array.
/// \tparam    S      The size of the array.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator/(T scalar, const Array<U, S>& a)
{
  auto result = a;
  unrolled_for_bounded<max_unroll_depth>([&] (auto i)
  {
    result[i] = scalar / result[i];
  });
  return result;
}

/// Overload of addition operator to perform elementwise addition of a scalar
/// constant to an array.
/// \param[in] scalar The scalar to multiply to each element of the array.
/// \param[in] a      The array to multiply with the scalar.
/// \tparam    T      The type of the scalar data.
/// \tparam    U      The type of the data in the array.
/// \tparam    S      The size of the array.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator+(T scalar, const Array<U, S>& a)
{
  auto result = a;
  unrolled_for_bounded<max_unroll_depth>([&] (auto i)
  {
    result[i] += scalar;
  });
  return result;
} 

/// Overload of subtraction operator to perform elementwise subtraction of a
/// constant and an array.
/// \param[in] scalar The scalar to multiply to each element of the array.
/// \param[in] a      The array to multiply with the scalar.
/// \tparam    T      The type of the scalar data.
/// \tparam    U      The type of the data in the array.
/// \tparam    S      The size of the array.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator-(T scalar, const Array<U, S>& a)
{
  auto result = a;
  unrolled_for_bounded<max_unroll_depth>([&] (auto i)
  {
    result[i] = scalar - result[i];
  });
  return result;
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_ARRAY_HPP