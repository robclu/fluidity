//==--- fluidity/container/host_tensor.hpp------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  host_tensor.hpp
/// \brief This file defines a file which implements host (CPU) side tensor
///        functionality where the tensor dimensions of the tensor are specified
///        at compile time.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_HOST_TENSOR_HPP
#define FLUIDITY_CONTAINER_HOST_TENSOR_HPP

#include "base_tensor.hpp"
#include "tensor_fwrd.hpp"
#include <fluidity/iterator/tensor_iterator.hpp>
#include <fluidity/utility/cuda.hpp>
#include <cstddef>

namespace fluid {

//==--- HostTensor 1D Specialization ---------------------------------------==//

/// Specialization for the case that the tensor is 1 Dimensional.
/// \tparam T The type of the data to store in the tensor.
template <typename T>
class HostTensor<T, 1> : public BaseTensor<T, 1> {
 private:
  /// Defines the device version of the tensor to be a friend of this class.
  template <typename TT, std::size_t D>
  friend class DeviceTensor;

 public:
  /// Defines the type of the tensor.
  using self_t            = HostTensor;
  /// Defines an alias for the base tensor class.
  using base_t            = BaseTensor<T, 1>;
  /// Defines the type of the elements in the tensor.
  using value_t           = typename base_t::value_t;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = typename base_t::pointer_t;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&; 
  /// Defines the type of a non const iterator.
  using iterator_t        = TensorIterator<self_t, false>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = TensorIterator<self_t, true>;

  /// Creates a host tensor with no elements. This requires the tensor to be
  /// resized before using it.
  HostTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D ensor.
  HostTensor(std::size_t elements);

  /// Cleans up any memory allocated for the tensor.
  ~HostTensor();

  /// Constructor to create a host tensor from a device tensor.
  /// \param[in] dev_tensor The device tensor to create the host tensor from.
  HostTensor(const DeviceTensor<T, 1>& dev_tensor);

  /// Returns an iterator to the first element in the tensor.
  iterator_t begin()
  {
    return iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  iterator_t end()
  {
    return iterator_t{this->_data + this->_size};
  }

  /// Resizes the tensor to contain \p num_elements elements.
  /// \param[in] num_elements The number of elements to resize the tensor to.
  void resize(std::size_t num_elements);

  /// Returns the size of the tensor for dimenison \p i. For this tensor
  /// implementation the dimension is ignored.
  /// \param[in] dim The dimension to get the size of.
  std::size_t size(std::size_t dim = 0) const;

 private:
  /// Allocates memory for the array.
  void allocate();

  /// Cleans up any memory allocated for the tensor.
  void cleanup();
};

//==--- BaseTensor 1D Implementation ---------------------------------------==//

//===== Public ----------------------------------------------------------=====//

template <typename T>
HostTensor<T, 1>::HostTensor(std::size_t elements) 
: BaseTensor<T, 1>(elements)
{
  allocate();
}

template <typename T>
HostTensor<T, 1>::~HostTensor()
{
  cleanup();
}

template <typename T>
HostTensor<T, 1>::HostTensor(const DeviceTensor<T, 1>& dev_tensor)
: BaseTensor<T, 1>(dev_tensor.size())
{
  allocate();
  util::cuda::memcpy_device_to_host(dev_tensor._data, this->_data, this->_size);
}

template <typename T>
void HostTensor<T, 1>::resize(std::size_t num_elements)
{
  cleanup();
  this->_size = num_elements;
  allocate();
}

template <typename T>
std::size_t HostTensor<T, 1>::size(std::size_t /*dim*/) const
{
  return this->_size;
}

//===== Private ---------------------------------------------------------=====//

template <typename T>
void HostTensor<T, 1>::allocate()
{
  // TODO: Add an implementation for aligned allocation...
  if (this->_data == nullptr) 
  {
    this->_data = static_cast<pointer_t>(malloc(this->mem_requirement()));
  }
}

template <typename T>
void HostTensor<T, 1>::cleanup()
{
  if (this->_data != nullptr)
  {
    free(this->_data);
  }
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_HOST_TENSOR_HPP