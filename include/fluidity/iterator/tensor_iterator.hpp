//==--- fluidity/iterator/tensor_iterator.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_iterator.hpp
/// \brief This file defines an iterator for tensors.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_TENSOR_ITERATOR_HPP
#define FLUIDITY_ITERATOR_TENSOR_ITERATOR_HPP

#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <type_traits>

namespace fluid {

/// The TensorIterator class conforms to the BidirectionalIterator concept, and
/// is a class which can iterate over tensors of 1, 2 and 3 dimensions.
/// \tparam T       The type of the tensor to iterator over.
/// \tparam IsConst If the iterator is a const iterator.
template <typename T, bool IsConst = false>
struct TensorIterator {
  /// Defines the type of the TensorIterator.
  using self_t       = TensorIterator;
  /// Defines the type of the elements the tensor holds, and which are iterated
  /// over.
  using element_t    = typename T::element_t;
  /// Defines the type of a reference to an element.
  using reference_t  = std::conditional_t<IsConst, const element_t&, element_t&>;
  /// Defines the type of the pointer for the iterator.
  using pointer_t    = std::conditional_t<IsConst, const element_t*, element_t*>;
  /// Defines the type of the difference between two iterators.
  using difference_t = int;

  /// Sets the element the iterator points to, and the offset to the next 
  fluidity_host_device TensorIterator(pointer_t ptr) : _ptr{ptr} {}

  /// Overload of copy constructor to copy an iterator.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device TensorIterator(const self_t& other) : _ptr{other._ptr} {}

  /// Moves the pointer from other to this tensor.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device TensorIterator(self_t&& other) : _ptr{other._ptr} {}

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
    return TensorIterator(self._ptr + offset);
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
    return TensorIterator(self._ptr + offset);
  }

  /// Overload of subtraction operator to compute the difference between two
  /// iterators, ie :$\f d = b - a \f$.
  /// \param[in] b The iterator to use as the start value.
  /// \param[in] a The iterator to subtract with.
  fluidity_host_device friend difference_t operator-(self_t b, self_t a)
  {
    return b._ptr - a._ptr;
  }

  /// Overload of copy assignment operator to copy the pointer the iterator
  /// is pointing to.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device self_t& operator=(const self_t& other)
  {
    _ptr = other._ptr; return *this;
  }

  /// Overload of move assignment operator -- this sets all the elements the
  /// iterator iterates over, to the values that the other iterator iterates
  /// over.
  /// \param[in] other The other iterator to iterate over.
  fluidity_host_device self_t& operator=(self_t&& other)
  {
    _ptr = other._ptr; return *this;
  }

  /// Overload of postfix operator to increment the iterator and return a new
  /// iterator to the next element.
  /// \param[in] junk Junk value to specify postfix.
  fluidity_host_device self_t operator++(int /*junk*/)
  { 
    self_t i = *this; move_pointer(1); return i; 
  } 

  /// Overload of prefix operator to increment the iterator and return a new
  /// iterator to the next element.
  /// \param[in] junk Junk value to specify postfix.
  fluidity_host_device self_t operator++()
  {
    move_pointer(1); return *this;
  }

  /// Overload of postdecrement operator to iecrement the iterator and return
  /// a new iterator to the previous element.
  /// \param[in] junk Junk value to specify post decrement.
  fluidity_host_device self_t operator--(int /*junk*/) 
  {
    self_t i = *this; move_pointer(-1); return i;
  }

  /// Overload of predecrement operator to decrement the iterator and return a
  /// new iterator to the previous element.
  /// \param[in] junk Junk value to specify postdecrement.
  fluidity_host_device self_t operator--()
  {
    move_pointer(-1); return *this;
  }  

  /// Overload of add and modify operation -- adds an \p offset to the pointer
  /// and returns a reference to the modified iterator.
  /// \param offset The offset to add to the iterator.
  fluidity_host_device self_t& operator+=(difference_t offset)
  { 
    move_pointer(offset); return *this;
  }

  /// Overload of subtract and modify operation -- subtracts an \p offset from
  /// the pointer and returns a reference to the modified iterator.
  /// \param offset The offset to subtract to the iterator.
  fluidity_host_device self_t& operator-=(difference_t offset)
  { 
    move_pointer(-offset); return *this;
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
  fluidity_host_device self_t operator[](difference_t offset) const
  {
    return *this + offset;
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
    return _ptr != other._ptr;
  }

 private:
  pointer_t _ptr; //!< The element the iterator points to.

  /// Moves the pointer to which the data points by \p amount.
  /// \param[in] amount The amount to move the data by.
  fluidity_host_device void move_pointer(int amount) 
  {
    _ptr += amount;
  }
};

//==--- Implementation -----------------------------------------------------==//

} // namespace fluid

#endif //FLUIDITY_ITERATOR_TENSOR_ITERATOR_HPP