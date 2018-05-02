//==--- fluidity/iterator/strided_iterator.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  strided_iterator.hpp
/// \brief This file defines an iterator which iterates with a specific stride.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_STRIDED_ITERATOR_HPP
#define FLUIDITY_ITERATOR_STRIDED_ITERATOR_HPP

#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/utility/debug.hpp>
#include <type_traits>

namespace fluid {

/// The StridedIterator class conforms to the BidirectionalIterator concept, and
/// is a class which can iterates between elements based on a stride. It can
/// therefore iterate over a single dimension in a multidimensional space by
/// setting the stride appropriately.
/// \tparam T       The type of the tensor to iterator over.
/// \tparam IsConst If the iterator is a const iterator.
/// \tparam Exec    The type of execution policy for the iterator -- what type
///                 of computational device it must use for execution.
template <typename T, bool IsConst = false, typename Exec = exec::default_type>
struct StridedIterator {
  /// Defines the type of the TensorIterator.
  using self_t        = StridedIterator;
  /// Defines the type of the elements the tensor holds.
  using value_t       = std::decay_t<T>;
  /// Defines the type of a reference to an element.
  using reference_t   = std::conditional_t<IsConst, const value_t&, value_t&>;
  /// Defines the type of the pointer for the iterator.
  using pointer_t     = std::conditional_t<IsConst, const value_t*, value_t*>;
  /// Defines the type of the difference between two iterators.
  using difference_t  = int;
  /// Defines the type of the size type for the stride.
  using stride_t      = difference_t;
  /// Defines the type of the execution policy for the iterator.
  using exec_t        = Exec;

  /// Defines the number of dimensions for the iterator.
  static constexpr std::size_t dimensions = 1;

  /// Sets the element the iterator points to, and the offset to the next 
  /// \param[in] ptr The pointer to iterate from.
  fluidity_host_device StridedIterator(pointer_t ptr) : _ptr{ptr} {}

  /// Sets the element the iterator points to, and the offset to the next 
  /// \param[in] ptr    The pointer to iterate from.
  /// \param[in] stride The stride between elements.
  fluidity_host_device StridedIterator(pointer_t ptr, stride_t stride)
  : _ptr{ptr}, _stride{stride} {}

  /// Overload of copy constructor to copy an iterator.
  /// \param[in] other The other iterator to iterate over.
  StridedIterator(const self_t& other) = default;

  /// Moves the pointer from other to this tensor.
  /// \param[in] other The other iterator to iterate over.
  StridedIterator(self_t&& other) = default;

  /// Overload of addition operator to add a difference_t and an iterator.
  /// For example:
  /// 
  /// \code{cpp}
  /// tensor_iterator_t t;
  /// 4 + t;                // Advances the iterator by 4 elements.
  /// \endcode
  /// 
  /// \param[in] offset The offset to add to the iterator.
  /// \param[in] self   The iterator to add to the difference_type.
  fluidity_host_device friend self_t
  operator+(difference_t offset, const self_t& self)
  {
    return self_t{self._ptr + offset * self._stride, self._stride};
  }

  /// Overload of subtraction operator to subtract an iterator from a
  /// difference_t. For example:
  /// 
  /// \code{cpp}
  /// tensor_iterator_t t;
  /// 4 - t;                // Reduces the iterator 4 elements.
  /// \endcode
  /// 
  /// \param[in] offset The offset to subtract the Iterator from.
  /// \param[in] self   The iterator to subtract with.
  fluidity_host_device friend self_t
  operator-(difference_t offset, const self_t& self)
  {
    return self_t{self._ptr - offset * self._stride, self.stride};
  }

  /// Overload of subtraction operator to compute the difference between two
  /// iterators, ie :$\f d = b - a \f$.
  /// \param[in] b The iterator to use as the start value.
  /// \param[in] a The iterator to subtract with.
  fluidity_host_device friend difference_t operator-(self_t b, self_t a)
  {
    // TODO: When debug is available:
    //  debug::log(debug::log::non_critical, [&]
    //  {
    //    if (a._stride != b._stride)
    //    {
    //      fluidity_debug_print(
    //        "Difference between iterators with different strides!");
    //    }
    //  });
    return (b._ptr - a._ptr) / a._stride;
  }

  /// Overload of copy assignment operator to copy the pointer the iterator
  /// is pointing to.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device self_t& operator=(const self_t& other)
  {
    _ptr = other._ptr; _stride = other._stride; return *this;
  }

  /// Overload of move assignment operator -- this sets all the elements the
  /// iterator iterates over, to the values that the other iterator iterates
  /// over.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device self_t& operator=(self_t&& other)
  {
    _ptr = other._ptr; _stride = other._stride; return *this;
  }

  /// Overload of postfix operator to increment the iterator and return a new
  /// iterator to the next element.
  /// \param[in] junk Junk value to specify postfix.
  fluidity_host_device self_t operator++(int /*junk*/)
  { 
    self_t i = *this; move_pointer(_stride); return i; 
  } 

  /// Overload of prefix operator to increment the iterator and return a new
  /// iterator to the next element.
  /// \param[in] junk Junk value to specify postfix.
  fluidity_host_device self_t operator++()
  {
    move_pointer(_stride); return *this;
  }

  /// Overload of postdecrement operator to iecrement the iterator and return
  /// a new iterator to the previous element.
  /// \param[in] junk Junk value to specify post decrement.
  fluidity_host_device self_t operator--(int /*junk*/) 
  {
    self_t i = *this; move_pointer(-_stride); return i;
  }

  /// Overload of predecrement operator to decrement the iterator and return a
  /// new iterator to the previous element.
  /// \param[in] junk Junk value to specify postdecrement.
  fluidity_host_device self_t operator--()
  {
    move_pointer(-_stride); return *this;
  }  

  /// Overload of add and modify operation -- adds an \p offset to the pointer
  /// and returns a reference to the modified iterator.
  /// \param offset The offset to add to the iterator.
  fluidity_host_device self_t& operator+=(difference_t offset)
  { 
    move_pointer(offset * _stride); return *this;
  }

  /// Overload of subtract and modify operation -- subtracts an \p offset from
  /// the pointer and returns a reference to the modified iterator.
  /// \param offset The offset to subtract to the iterator.
  fluidity_host_device self_t& operator-=(difference_t offset)
  { 
    move_pointer(-(offset * _stride)); return *this;
  }

  /// Overload of addition operator -- adds an offset to the pointer and
  /// returns a new iterator to the element at this iterator's pointer plus
  /// \p offset.
  /// \param[in] offset The offset to add to the iterator.
  fluidity_host_device self_t operator+(difference_t offset) const
  {
    self_t other = *this; return other += offset;
  }

  /// Overload of subtraction operator -- subtracts an offset from the pointer
  /// and returns a new iterator to the element at this iterator's pointer
  /// element minus \p offset.
  /// \param[in] offset The offset to subtract from the iterator.
  fluidity_host_device self_t operator-(difference_t offset) const
  {
    self_t other = *this; return other -= offset;
  }


  /// Overload of indirection operator to return a reference to the element.
  /// Returns a reference to the data the iterator currently points to.
  fluidity_host_device reference_t operator*() const
  {
    return *_ptr;
  }

  /// Overload of dereference operator to return the element pointer.
  /// Returns the pointer to the data the iterator currently points to.
  fluidity_host_device pointer_t operator->() const
  {
    return _ptr;
  }

  /// Overload of access operator to return the value of a specific element
  /// the iterator iterates over, at \p offset from the starting element to
  /// iterate over. For example:
  /// 
  /// \code{cpp}
  /// tensor_iterator_t t;
  /// 
  /// // new_iterator is a tensor_iterator_t pointer to the element 4 positions
  /// // from the element t points to.
  /// auto new_iterator = t[4];
  /// \endcode
  /// 
  /// \param[in] offset The offset of the iterator to access, from this
  ///                   iterator.
  //fluidity_host_device self_t operator[](difference_t offset) const
  //{
  //  return *this + offset * _stride;
  //}

  /// Overload of access operator to return the value of a specific element
  /// the iterator iterates over, at \p offset from the starting element to
  /// iterate over. For example:
  /// 
  /// \code{cpp}
  /// tensor_iterator_t t;
  /// 
  /// // new_iterator is a tensor_iterator_t pointer to the element 4 positions
  /// // from the element t points to.
  /// auto new_iterator = t[4];
  /// \endcode
  /// 
  /// \param[in] offset The offset of the iterator element to access, from this
  ///                   iterator.
  fluidity_host_device value_t& operator[](difference_t offset)
  {
    return *(_ptr + offset * _stride);
  }

  /// Tests if one iterator is equal to another iterator (when they both point
  /// to the same elements).
  /// \param[in] other The other iterator to compare against.
  fluidity_host_device bool operator==(const self_t& other) const
  {
    return _ptr == other._ptr;
  }

  /// Tests if one iterator is not equal to another iterator (when they both
  /// point to different elements).
  /// \param[in] other The other iterator to compare against.
  fluidity_host_device bool operator!=(const self_t& other) const
  {
    return _ptr != other._ptr || _stride != other._stride;
  }

  /// Offsets the iterator by \p amount and returns a new iterator to the offset
  /// element.
  /// \param[in]  amount  The amount of offset from this iterator.
  /// \tparam     Value   The value which defines the dimension.
  fluidity_host_device self_t offset(stride_t amount) const
  {
    return self_t{_ptr + amount * _stride, _stride};
  }

  /// Shifts the iterator by \p amount, modifying the iterator. This can shift
  /// the iterator forward and backward based on the sign of \p amount
  /// (+ = forward).
  /// \param[in]  amount  The amount to advance the iterator by.
  fluidity_host_device constexpr self_t& shift(stride_t amount)
  {
    _ptr += amount * _stride; return *this;
  }

  /// Returns the number of dimensions which can be iterated over.
  fluidity_host_device constexpr std::size_t num_dimensions() const
  {
    return 1;
  }

  /// Returns the backward difference between this iterator and the iterator \p
  /// amount places from from this iterator. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{i} - U_{i - \texrm{amount}}
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->backward_diff();
  /// auto diff = state->backward_diff(1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  fluidity_host_device value_t backward_diff(unsigned int amount = 1) const
  {
    return *_ptr - *(_ptr - amount * _stride);
  }

  /// Returns the forward difference between this iterator and the iterator \p
  /// amount places from from this iterator. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{i + amount} - U_{i}
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->forward_diff();
  /// auto diff = state->forward_diff(1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  fluidity_host_device value_t forward_diff(unsigned int amount = 1) const
  {
    return *(_ptr + amount * _stride) - *_ptr;
  }

  /// Returns the central difference between the iterator \p amount places
  /// forward from this iterator and the iterator \p amount places backward
  /// from this iterator. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{i + amount} - U_{i - amount}
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->central_diff();
  /// auto diff = state->central_diff(1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  fluidity_host_device value_t central_diff(unsigned int amount = 1) const
  {
    const auto shift = amount * _stride;
    return *(_ptr + shift) - *(_ptr - shift);
  }

 private:
  pointer_t _ptr    = nullptr;  //!< The element the iterator points to.
  stride_t  _stride = 1;        //!< The stride between elements.

  /// Moves the pointer to which the data points by \p amount.
  /// \param[in] amount The amount to move the data by.
  fluidity_host_device void move_pointer(stride_t amount) 
  {
    _ptr += amount;
  }
};

//==--- Implementation -----------------------------------------------------==//

} // namespace fluid

#endif //FLUIDITY_ITERATOR_TENSOR_ITERATOR_HPP