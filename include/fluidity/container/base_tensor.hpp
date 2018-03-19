//==--- fluidity/container/base_tensor.hpp------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  base_tensor.hpp
/// \brief This file defines the functionality which is common to both the host
///        and the device tensor implementations.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_BASE_TENSOR_HPP
#define FLUIDITY_CONTAINER_BASE_TENSOR_HPP

#include <fluidity/utility/portability.hpp>
#include <cstddef>
#include <type_traits>

namespace fluid {

/// Implementation of a base tensor class which contains functionality common
/// to both host and device implementations of the tensor class.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
class BaseTensor {
 public:
  /// Defines the type of the data being stored in the tensor.
  using value_t           = std::decay_t<T>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = value_t*;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&;

  /// Creates a base tensor, with no elements or size. Calling this constructor
  /// required resizing the tensor.
  /*fluidity_host_device*/ BaseTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D tensor.
  fluidity_host_device BaseTensor(std::size_t elements);

  /// Returns a reference to the element at position i in the tensor. This
  /// is independent of the dimensionality of the tensor, it treads the tensor
  /// as flattened.
  /// \param[in] i The index of the element to get.
  fluidity_host_device reference_t operator[](int i)
  {
    return _data[i];
  }

  /// Returns a const reference to the element at position i in the tensor. This
  /// is independent of the dimensionality of the tensor, it treads the tensor
  /// as flattened.
  /// \param[in] i The index of the element to get.
  fluidity_host_device const_reference_t operator[](int i) const
  {
    return _data[i];
  }

  /// Returns the amount of memory required by the tensor, in bytes.
  fluidity_host_device std::size_t mem_requirement() const;

  /// Returns the total number of elements in the tensor.
  fluidity_host_device std::size_t total_size() const;

 protected:
  pointer_t   _data = nullptr;  //!< Pointer to the tensor data.
  std::size_t _size = 0;        //!< Number of elements in the tensor.
};

//==--- BaseTensor Implementation ------------------------------------------==//

template <typename T, std::size_t D>
BaseTensor<T, D>::BaseTensor(std::size_t elements)
: _data(nullptr), _size(elements) {}

template <typename T, std::size_t D>
std::size_t BaseTensor<T, D>::mem_requirement() const
{
  return sizeof(value_t) * _size;
}

template <typename T, std::size_t D>
std::size_t BaseTensor<T, D>::total_size() const
{
  return _size;
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_BASE_TENSOR_HPP