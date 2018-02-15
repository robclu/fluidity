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

namespace fluid {

/// Implementation of a host tensor class which specializes the components of
/// the tensor implementation which are specific to the host side.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
class HostTensor {
 public:

};

//==--- HostTensor 1D Specialization ---------------------------------------==//

/// Specialization for the case that the tensor is 1 Dimensional.
/// \tparam T The type of the data to store in the tensor.
template <typename T>
class HostTensor<T, 1> : public BaseTensor<T, 1> {
 public:
  /// Defines an alias for the base tensor class.
  using base_t    = BaseTensor<T, 1>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t = typename base_t::pointer_t;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D ensor.
  HostTensor(std::size_t elements);

  /// Cleans up any memory allocated for the tensor.
  ~HostTensor();

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
:   BaseTensor<T, 1>(elements) {
  allocate();
}

template <typename T>
HostTensor<T, 1>::~HostTensor() {
  cleanup();
}

//===== Private ---------------------------------------------------------=====//

template <typename T>
void HostTensor<T, 1>::allocate() {
  // TODO: Add an implementation for aligned allocation...
  if (this->_data == nullptr) {
    this->_data = static_cast<pointer_t>(malloc(this->mem_requirement()));
  }
}

template <typename T>
void HostTensor<T, 1>::cleanup() {
  if (this->_data != nullptr) {
    free(this->_data);
  }
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_HOST_TENSOR_HPP