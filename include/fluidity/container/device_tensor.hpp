//==--- fluidity/container/device_tensor.hpp---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_tensor.hpp
/// \brief This file defines a file which implements device (GPU) side tensor
///        functionality where the tensor dimensions of the tensor are specified
///        at compile time.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_DEVICE_TENSOR_HPP
#define FLUIDITY_CONTAINER_DEVICE_TENSOR_HPP

#include "base_tensor.hpp"
#include "tensor_fwrd.hpp"
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/tensor_iterator.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid {

//==--- HostTensor 1D Specialization ---------------------------------------==//

/// Specialization for the case that the tensor is 1 Dimensional.
/// \tparam T The type of the data to store in the tensor.
template <typename T>
class DeviceTensor<T, 1> : public BaseTensor<T, 1> {
 private:
  /// Defines the host version of the tensor to be a friend of this class.
  template <typename TT, std::size_t D>
  friend class HostTensor;

  /// Defines the type of the executor for the iterator to be a GPU executor.
  using exec_t = exec::gpu_type;

 public:
  /// Defines the type of the tensor.
  using self_t            = DeviceTensor;
  /// Defines the type of the elements in the tensor.
  using element_t         = T;
  /// Defines an alias for the base tensor class.
  using base_t            = BaseTensor<T, 1>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = typename base_t::pointer_t;
  /// Defines the type of a non const iterator.
  using iterator_t        = TensorIterator<self_t, false, exec_t>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = TensorIterator<self_t, true, exec_t>;

  /// Defines the number of dimensions in the tensor

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D ensor.
  DeviceTensor(std::size_t elements);

  /// Cleans up any memory allocated for the tensor.
  ~DeviceTensor();

  /// Constructor to create a device tensor from a host tensor.
  /// \param[in] host_tensor The host tensor to create the device tensor from.
  HostTensor(const HostTensor<T, 1>& host_tensor);

  /// Returns an iterator to the first element in the tensor.
  fluidity_host_device iterator_t begin()
  {
    return iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  fluidity_host_device iterator_t end()
  {
    return iterator_t{this->_data + this->_size};
  }

  /// Returns an iterator to the first element in the tensor.
  fluidity_host_device const_iterator_t begin() const
  {
    return const_iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  fluidity_host_device const_iterator_t end() const
  {
    return const_iterator_t{this->_data + this->_size};
  }

  /// Resizes the tensor to contain \p num_elements elements.
  /// \param[in] num_elements The number of elements to resize the tensor to.
  void resize(std::size_t num_elements);

  /// Returns the size of the tensor for dimenison \p i. For this tensor
  /// implementation the dimension is ignored.
  /// \param[in] dim The dimension to get the size of.
  fluidity_host_device std::size_t size(std::size_t dim = 0) const;

 private:
  bool _must_free = true; //!< Sets if the memory must be freed.

  /// Allocates memory for the array.
  void allocate();

  /// Cleans up the memory allocated for the tensor.
  void cleanup();
};

//==--- DeviceTensor 1D Implementation -------------------------------------==//

//===== Public ----------------------------------------------------------=====//

template <typename T>
DeviceTensor<T, 1>::DeviceTensor(std::size_t elements)
: BaseTensor<T, 1>(elements)
{
  allocate();
}

template <typename T>
DeviceTensor<T, 1>::~DeviceTensor()
{
  cleanup();
}

template <typename T>
DeviceTensor<T, 1>::DeviceTensor(const HostTensor<T, 1>& host_tensor)
: BaseTensor<T, 1>(host_tensor.size())
{
  allocate();
  util::cuda::memcpy_host_to_device(host_tensor._data      ,
                                    this->_data            ,
                                    this->mem_requirement());
}

template <typename T>
void DeviceTensor<T, 1>::resize(std::size_t num_elements)
{
  cleanup();
  this->_size = num_elements;
  allocate();
}

template <typename T>
std::size_t DeviceTensor<T, 1>::size(std::size_t /*dim*/) const
{
  return this->_size;
}

//===== Private ---------------------------------------------------------=====//

template <typename T>
void DeviceTensor<T, 1>::allocate()
{
  // TODO: Add an implementation for aligned allocation...
  if (this->_data == nullptr) 
  {
    util::cuda::allocate(reinterpret_cast<void**>(&this->_data),
                         this->mem_requirement());
  }
}

template <typename T>
void DeviceTensor<T, 1>::cleanup()
{
  if (this->_data != nullptr && this->_must_free)
  {
    util::cuda::free(this->_data);
  }
}


} // namespace fluid

#endif // FLUIDITY_CONTAINER_DEVICE_TENSOR_HPP