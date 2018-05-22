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

  /// The default constructor uses the default initialization.
  constexpr Array() = default;

  /// Initializes each of the elements in the array to have the value \p value.
  /// \param[in] value The value to set the array elements to.
  fluidity_host_device constexpr Array(value_t value)
  {
    fill(this->begin(), this->end(), value);
  }

  /// Initializes the elements in the array to have the values \p values.
  /// \param[in] values The values to set the array elements to.
  /// \tparam    Values The types of the values.
  template <typename... Values>
  fluidity_host_device constexpr Array(Values&&... values)
  : _data{ static_cast<value_t>(values)... } {}

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
  /// element in the array.
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

  /// Overload of operator- to add each element of a container to each element
  /// in the array.
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
};

//==--- Array Implementation -----------------------------------------------==//

//===== Operator --------------------------------------------------------=====//

template < typename    T
         , std::size_t S
         , std::enable_if_t<(S < max_unroll_depth), int> = 0>
fluidity_host_device constexpr auto operator*(T scalar, const Array<T, S>& a)
{
  auto result = a;
  unrolled_for<S>([&] (auto i)
  {
    result[i] *= scalar;
  });
  return result;
} 

template < typename    T
         , std::size_t S
         , std::enable_if_t<!(S < max_unroll_depth), int> = 0>
fluidity_host_device constexpr auto operator*(T scalar, const Array<T, S>& a)
{
  auto result = a;
  for (auto i : range(S))
  {
    result[i] *= scalar;
  }
  return result;
} 

//===== Public ----------------------------------------------------------=====//

//===== Private ---------------------------------------------------------=====//


} // namespace fluid

#endif // FLUIDITY_CONTAINER_ARRAY_HPP