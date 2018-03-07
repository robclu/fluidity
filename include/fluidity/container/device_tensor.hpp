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
#include <fluidity/utility/debug.hpp>

namespace fluid {

/// Implementation of a device tensor class which specializes the components of
/// the tensor implementation which are specific to the device side.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
class DeviceTensor {
 public:

};

//==--- HostTensor 1D Specialization ---------------------------------------==//

template <typename T>

/// Specialization for the case that the tensor is 1 Dimensional.
/// \tparam T The type of the data to store in the tensor.
template <typename T>
class DeviceTensor<T, 1> : public BaseTensor<T, 1> {
 public:
  /// Defines the type of the elements in the tensor.
  using element_t = T;
  /// Defines an alias for the base tensor class.
  using base_t    = BaseTensor<T, 1>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t = typename base_t::pointer_t;

  /// Defines the number of dimensions in the tensor

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D ensor.
  fluidity_host_device DeviceTensor(std::size_t elements);

  /// Cleans up any memory allocated for the tensor.
  fluidity_host_device ~DeviceTensor();

  /// Resizes the tensor to contain \p num_elements elements.
  /// \param[in] num_elements The number of elements to resize the tensor to.
  fluidity_host_device void resize(std::size_t num_elements);

  /// Returns the size of the tensor for dimenison \p i. For this tensor
  /// implementation the dimension is ignored.
  /// \param[in] dim The dimension to get the size of.
  fluidity_host_device std::size_t size(std::size_t /*dim*/) const;

 private:
  bool _must_free = true; //!< Sets if the memory must be freed.

  /// Allocates memory for the array.
  fluidity_host_device void allocate();

  /// Cleans up the memory allocated for the tensor.
  fluidity_host_device void cleanup();
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
    fluidity_check_cuda_result(
      cudaMalloc(reinterpret_cast<void**>(&this->_data), 
                 this->mem_requirement())
    );
  }
}

template <typename T>
void DeviceTensor<T, 1>::cleanup()
{
  if (this->_data != nullptr && this->_must_free)
  {
    fluidity_check_cuda_result(cudaFree(this->_data));
  }
}


} // namespace fluid

#endif // FLUIDITY_CONTAINER_DEVICE_TENSOR_HPP