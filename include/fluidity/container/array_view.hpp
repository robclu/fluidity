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

  /// \todo Implement StidedIterator and change this to use those.
  
  /// Defines the type of a non const iterator.
  //using iterator_t        = StridedIterator<self_t, false>;
  /// Defines the type of a const iterator.
  //using const_iterator_t  = StridedIterator<self_t, true>;

  /// Initializes pointer to the first element and the step size.
  /// \param[in] ptr  The pointer to the first element.
  /// \param[in] step The step size between elements that are being referenced.
  fluidity_host_device ArrayRef(pointer_t ptr, std::size_t step)
  : _ptr(ptr), _step(step) {}

  /// Creates a new array view by setting the pointer and the step size to that
  /// of the other array view.
  /// \param[in] other The other ArrayRef to copy.
  fluidity_host_device ArrayRef(const self_t& other)
  : _ptr(other._ptr), _step(other._step) {}

  /// Creates a new array view by setting the pointer and the step size to that
  /// of the other array view. This moves the pointer from \p other to this
  /// array view, invalidating \p other.
  /// \param[in] other The other ArrayView to copy.
  fluidity_host_device ArrayRef(self_t&& other) noexcept
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
  fluidity_host_device const_reference_t operator[](SizeType i) const
  {
    return _ptr[i * _step];
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

  /// Defines the 

  /// Copies the contents of a container into the contents which are viewed by
  /// this class.
  /// \param[in] container The container to copy the data from.
  /// \tparam    Container The type of the container.
  template <typename Container>
  void copy_from_container(Container&& container);
};

//==--- Implementation -----------------------------------------------------==//

//===== Private ---------------------------------------------------------=====//

template <typename T, std::size_t Elements> template <typename Container>
void ArrayView<T, Elements>::copy_from_container(Container&& container)
{
  if constexpr (size() < max_unroll_depth)
  {
    unrolled_for<size()>([this] (auto i)
    {
      _ptr[i * _step] = container[i];
    });
  }
  else
  {
    for (auto i : range(size()))
    {
      _ptr[i * step] = container[i];
    }
  }
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_ARRAY_VIEW_HPP