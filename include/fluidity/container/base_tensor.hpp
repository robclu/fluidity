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

};

//==--- BaseTensor 1D Specialization ---------------------------------------==//

/// Specialization for the case that the tensor is 1 Dimensional.
/// \tparam T The type of the data to store in the tensor.
template <typename T>
class BaseTensor<T, 1> {
 public:
  /// Defines the type of the data being stored in the tensor.
  using value_t   = std::decay_t<T>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t = value_t*;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D tensor.
  fluidity_host_device BaseTensor(std::size_t elements);

  /// Returns the amount of memory required by the tensor, in bytes.
  fluidity_host_device std::size_t mem_requirement() const;

  /// Returns the number of elements in the tensor.
  fluidity_host_device std::size_t size() const;

 protected:
  pointer_t   _data = nullptr;  //!< Pointer to the tensor data.
  std::size_t _size = 0;        //!< Number of elements in the tensor.
};

//==--- BaseTensor 1D Implementation ---------------------------------------==//

template <typename T>
BaseTensor<T, 1>::BaseTensor(std::size_t elements)
: _data(nullptr), _size(elements) {}

template <typename T>
std::size_t BaseTensor<T, 1>::mem_requirement() const
{
  return sizeof(value_t) * _size;
}

template <typename T>
std::size_t BaseTensor<T, 1>::size() const
{
  return _size;
}


} // namespace fluid

#endif // FLUIDITY_CONTAINER_BASE_TENSOR_HPP