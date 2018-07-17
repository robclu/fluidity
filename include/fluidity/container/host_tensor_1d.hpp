//==--- fluidity/container/host_tensor_1d.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  host_tensor_1d.hpp
/// \brief This file defines a file which implements host (CPU) side tensor
///        functionality for a single dimension.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_HOST_TENSOR_1D_HPP
#define FLUIDITY_CONTAINER_HOST_TENSOR_1D_HPP

#include "base_tensor.hpp"
#include "tensor_fwrd.hpp"
#include "device_tensor.hpp"
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/iterator/strided_iterator.hpp>
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

  /// Defines the type of the executor for the iterator to be a GPU executor.
  using exec_t = exec::cpu_type;

 public:
  /// Defines the number of dimensions in the tensor.
  static constexpr auto dimensions = std::size_t{1};

  /// Defines the type of the tensor.
  using self_t            = HostTensor;
  /// Defines the type of the elements stored in the tensor.
  using element_t         = std::decay_t<T>;
  /// Defines an alias for the base tensor class.
  using base_t            = BaseTensor<element_t, 1>;
  /// Defines the type of the elements in the tensor.
  using value_t           = typename base_t::value_t;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = typename base_t::pointer_t;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&; 
  /// Defines the type of a non const iterator.
  using iterator_t        = StridedIterator<element_t, false, exec_t>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = StridedIterator<element_t, true, exec_t>;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<1>;
  /// Defines the type of a non const iterator.
  using multi_iterator_t  = MultidimIterator<element_t, dim_info_t, exec_t>;

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

  /// Constructor to copy a host tensor to another host tensor.
  /// \param[in] other The other host tensor to copy from.
  HostTensor(const self_t& other);

  /// Constructor to move a host tensor to another host tensor.
  /// \param[in] other The other host tensor to move from.
  HostTensor(self_t&& other);

  /// Overload of operator= to copy a host tensor to another host tensor.
  self_t& operator=(const self_t& other);

  /// Overload of operator= to move a host tensor to another host tensor.
  self_t& operator=(self_t&& other);

  /// Returns the HostTensor as a device tensor.
  DeviceTensor<T, 1> as_device() const;

  /// Returns an iterator to the first element in the tensor.
  iterator_t begin()
  {
    return iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  iterator_t end()
  {
    return iterator_t{this->_data + this->total_size()};
  }

  /// Returns an iterator to the first element in the tensor.
  const_iterator_t begin() const
  {
    return const_iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  const iterator_t end() const
  {
    return const_iterator_t{this->_data + this->total_size()};
  }

  /// Returns a multi dimensional iterator over the tensor data, which is
  /// initialized to point to the start of the tensor data. For a 1D tensor the
  /// multidimensional iterator behaves the same as a StridedIterator.
  fluidity_host_device multi_iterator_t multi_iterator() const
  {
    return multi_iterator_t{this->_data, this->total_size()};
  }

  /// Resizes the tensor to contain \p num_elements elements.
  /// \param[in] num_elements The number of elements to resize the tensor to.
  void resize(std::size_t num_elements);

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
HostTensor<T, 1>::HostTensor(const self_t& other) 
: BaseTensor<T, 1>(other.total_size())
{
  allocate();
  for (const auto i : range(other.total_size()))
  {
    this->_data[i] = other[i];
  }
}

template <typename T>
HostTensor<T, 1>::HostTensor(self_t&& other) 
{
  this->_data         = other._data;
  this->_dim_sizes[0] = other._dim_sizes[0];
  other._data         = nullptr;
  other._dim_sizes[0] = 0;
}

template <typename T>
HostTensor<T, 1>::HostTensor(const DeviceTensor<T, 1>& dev_tensor)
{
  this->_dim_sizes[0] = dev_tensor._dim_sizes[0];
  allocate();
  util::cuda::memcpy_device_to_host(dev_tensor._data       ,
                                    this->_data            ,
                                    this->mem_requirement());
}

template <typename T>
HostTensor<T, 1>& HostTensor<T, 1>::operator=(const self_t& other)
{
  this->_dim_sizes[0] = other._dim_sizes[0];
  allocate();
  for (const auto i : range(other.total_size()))
  {
    this->_data[i] = other[i];
  }
  return *this;
}

template <typename T>
HostTensor<T, 1>& HostTensor<T, 1>::operator=(self_t&& other) 
{
  this->_data         = other._data;
  this->_dim_sizes[0] = other._dim_sizes[0];
  other._data         = nullptr;
  other._dim_sizes[0] = 0;
  return *this;
}

template <typename T>
DeviceTensor<T, 1> HostTensor<T, 1>::as_device() const
{
  return DeviceTensor<T, 1>(*this);
}

template <typename T>
void HostTensor<T, 1>::resize(std::size_t num_elements)
{
  cleanup();
  this->_dim_sizes[0] = num_elements;
  allocate();
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

#endif // FLUIDITY_CONTAINER_HOST_TENSOR_1D_HPP