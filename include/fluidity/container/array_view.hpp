//==--- fluidity/container/array_view.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array_view.hpp
/// \brief This file defines the implementation of an array class which does not
///        own the underlying data, but rather just views it.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_ARRAY_VIEW_HPP
#define FLUIDITY_CONTAINER_ARRAY_VIEW_HPP

#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/iterator/range.hpp>

namespace fluid {

/// The ArrayView class allows portions of data to be viewed as an array. This
/// can view portions of the data in row major or column major (in >= 2 dims)
/// by modifying the stride of the view to match the number of elements in a
/// given direction of the container which is being viewed.
/// \tparam T         The type of the data to store in the tensor.
/// \tparam Elements  The number of elements in the array.
template <typename T, std::size_t Elements>
class ArrayView;

namespace detail {

/// Struct which can be used to check if the type T is an ArrayView type.
/// \tparam T The type to check.
template <typename T>
struct IsArrayView {
  /// Defines that this type is not an array view.
  static constexpr auto value = false;
};

/// Specialization for types which are ArrayView types.
/// \tparam T The type held by the ArrayView.
/// \tparam S The size of the ArrayView.
template <typename T, std::size_t S>
struct IsArrayView<ArrayView<T, S>> {
  /// Defines that this type is an array view.
  static constexpr auto value = true;
};

} // namespace detail

/// Returns true if the type T is an ArrayView.
/// \tparam T The type to check if is an ArrayView.
template <typename T>
static constexpr auto is_array_view_v = detail::IsArrayView<T>::value;

/// Implementation of the ArrayView class.
/// \tparam T         The type of the data to store in the tensor.
/// \tparam Elements  The number of elements in the array.
template <typename T, std::size_t Elements>
class ArrayView {
  /// Defines the type of the tensor.
  using self_t            = ArrayView;
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

  /// Defines a tag class which can be used to define unrolled and non-unrolled
  /// implementations of the array functionality.
  template <bool> struct UnrollTag {};

  /// Defines the type of tag for unrolled implementations.
  using unroll_true_tag_t  = UnrollTag<true>;
  /// Defines the type of a tage for non-unrolled implementations.
  using unroll_false_tag_t = UnrollTag<false>;

  /// Defines an instance of the unroll tag for this class based on the number
  /// of elements in the array.
  static constexpr auto unroll_tag = UnrollTag<(Elements < max_unroll_depth)>{};

 public:

  /// Initializes pointer to the first element and the step size.
  /// \param[in] ptr  The pointer to the first element.
  /// \param[in] step The step size between elements that are being referenced.
  fluidity_host_device ArrayView(pointer_t ptr, std::size_t step)
  : _ptr(ptr), _step(step) {}

  /// Creates a new array view by setting the pointer and the step size to that
  /// of the other array view.
  /// \param[in] other The other ArrayRef to copy.
  fluidity_host_device ArrayView(const self_t& other)
  : _ptr(other._ptr), _step(other._step) {}

  /// Creates a new array view by setting the pointer and the step size to that
  /// of the other array view. This moves the pointer from \p other to this
  /// array view, invalidating \p other.
  /// \param[in] other The other ArrayView to copy.
  fluidity_host_device ArrayView(self_t&& other)
  : _ptr(std::move(other._ptr)), _step(other._step)
  { 
    other._ptr = nullptr; other._step = 0;
  }

  /// Overload of access operator to return a reference to the \p ith element
  /// in the array.
  /// \param[in] i The index of the element to get a referemce to.
  fluidity_host_device reference_t operator[](std::size_t i)
  {
    return _ptr[i * _step];
  }

  /// Overload of access operator to return a constant reference to the
  /// \p ith element in the array.
  /// \param[in] i The index of the element to get a referemce to.
  /// \return    A constant reference to the element at position \p i.
  fluidity_host_device const_reference_t operator[](std::size_t i) const
  {
    return _ptr[i * _step];
  }

  /// Overload of operator- to subtract each element of a container from each
  /// element in the array view, returning a new array.
  /// \param[in] container The container to elementwise subtract from the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr auto operator-(Container&& container) const
  {
    auto result = Array<value_t, Elements>();
    unrolled_for_bounded<Elements>([&] (auto i)
    {
      result[i] = this->operator[](i) - container[i];
    });
    return result;
  }

  /// Overload of operator+ to add each element of a container to each element
  /// in the array.
  /// \param[in] container The container to elementwise add with the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr auto operator+(Container&& container) const
  {
    auto result = Array<value_t, Elements>();
    unrolled_for_bounded<Elements>([&] (auto i)
    {
      result[i] = this->operator[](i) + container[i];
    });
    return result;
  }

  /// Overload of operator-= to subtract each element of a container from each
  /// element in the array view, returning the modified array view.
  /// \param[in] container The container to elementwise subtract from the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t& operator-=(Container&& container)
  {
    unrolled_for_bounded<Elements>([&] (auto i)
    {
      this->operator[](i) -= container[i];
    });
    return *this;
  }

  /// Overload of operator+= to add each element of a container to each element
  /// in the array view, returning the modified array view.
  /// \param[in] container The container to elementwise add with the array.
  /// \tparam    Container The type of the container.
  template <typename Container>
  fluidity_host_device constexpr self_t& operator+=(Container&& container)
  {
    unrolled_for_bounded<Elements>([&] (auto i)
    {
      this->operator[](i) += container[i];
    });
    return *this;
  }

  /// Returns an iterator to the first element in the array.
  fluidity_host_device constexpr iterator_t begin()
  {
    return iterator_t{_ptr, _step};
  }

  /// Returns an iterator to the last element in the array.
  fluidity_host_device constexpr iterator_t end()
  {
    return iterator_t{_ptr + _step * Elements, _step};
  }

  /// Returns a constant iterator to the first element in the array.
  fluidity_host_device constexpr const_iterator_t begin() const
  {
    return const_iterator_t{_ptr, _step};
  }

  /// Returns a constant iterator to the last element in the array.
  fluidity_host_device constexpr const_iterator_t end() const
  {
    return const_iterator_t{_ptr + _step * Elements, _step};
  }

  /// Returns the number of elements in the array.
  fluidity_host_device constexpr std::size_t size() const
  {
    return Elements;
  }

  /// Returns the number of elements in the array.
  fluidity_host_device std::size_t step() const
  {
    return _step;
  }

 private:
  pointer_t   _ptr;   //!< Pointer to the data to view.
  std::size_t _step;  //!< Step size when moving between elements.

  /// Copies the contents of a container into the contents which are viewed by
  /// this class.
  /// \param[in] container The container to copy the data from.
  /// \tparam    Container The type of the container.
  template <typename Container>
  void copy_from_container(Container&& container);
};

//==--- Implementation -----------------------------------------------------==//

//===== Operator --------------------------------------------------------=====//

/// Overload of multiplication operator to perform elementwise multiplication of
/// a scalar constant to an array view. This returns am array, to modify the
/// contents of the array view, use *=.
/// \param[in] s  The scalar to multiply to each element of the array view.
/// \param[in] a  The array view to multiply with the scalar.
/// \tparam    T  The type of the scalar data.
/// \tparam    U  The type of the data in the array view.
/// \tparam    S  The size of the array view.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator*(T s, const ArrayView<U, S>& a)
{
  auto result = Array<U, S>();
  unrolled_for_bounded<S>([&] (auto i)
  {
    result[i] = s * a[i];
  });
  return result;
} 

/// Overload of division operator to perform elementwise division of
/// a scalar constant to an array view. This returns an array, to modify the
/// contents of the array view, use /=.
/// \param[in] s  The scalar to devide by each element.
/// \param[in] a  The array view divide with the scalar.
/// \tparam    T  The type of the scalar data.
/// \tparam    U  The type of the data in the array view.
/// \tparam    S  The size of the array view.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator/(T s, const ArrayView<U, S>& a)
{
  auto result = Array<U, S>();
  unrolled_for_bounded<S>([&] (auto i)
  {
    result[i] = s / a[i];
  });
  return result;
}

/// Overload of addition operator to perform elementwise addition of
/// a scalar constant to an array view. This returns an array, to modify the
/// contents of the array view, use +=.
/// \param[in] s  The scalar to add to each element of the array view.
/// \param[in] a  The array view to add with the scalar.
/// \tparam    T  The type of the scalar data.
/// \tparam    U  The type of the data in the array view.
/// \tparam    S  The size of the array view.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator+(T s, const ArrayView<U, S>& a)
{
  auto result = Array<U, S>();
  unrolled_for_bounded<S>([&] (auto i)
  {
    result[i] = s + a[i];
  });
  return result;
}

/// Overload of subtraction operator to perform elementwise subtraction of
/// a scalar constant to an array view. This returns an array, to modify the
/// contents of the array view, use -=.
/// \param[in] s  The scalar to add to each element of the array view.
/// \param[in] a  The array view to add with the scalar.
/// \tparam    T  The type of the scalar data.
/// \tparam    U  The type of the data in the array view.
/// \tparam    S  The size of the array view.
template <typename T, typename U, std::size_t S, conv_enable_t<T, U> = 0>
fluidity_host_device constexpr auto operator-(T s, const ArrayView<U, S>& a)
{
  auto result = Array<U, S>(s);
  unrolled_for_bounded<S>([&] (auto i)
  {
    result[i] = s - a[i];
  });
  return result;
} 

//===== Private ---------------------------------------------------------=====//

template <typename T, std::size_t Elements> template <typename Container>
void ArrayView<T, Elements>::copy_from_container(Container&& container)
{
  unrolled_for_bounded<Elements>([&] (auto i)
  {
    _ptr[i * _step] = container[i];
  });
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_ARRAY_VIEW_HPP