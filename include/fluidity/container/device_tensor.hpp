//==--- fluidity/container/device_tensor.hpp --------------- -*- C++ -*- ---==//
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
///        functionality for N dimensions.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_DEVICE_TENSOR_ND_HPP
#define FLUIDITY_CONTAINER_DEVICE_TENSOR_ND_HPP

#include "base_tensor.hpp"
#include "tensor_fwrd.hpp"
#include "host_tensor.hpp"
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/iterator/strided_iterator.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid {

/// Implementation of a device side tensor for N dimensions.
/// \tparam T The type of the data to store in the tensor.
/// \tparam N The number of dimensions for the tensor.
template <typename T, std::size_t N>
class DeviceTensor : public BaseTensor<T, N> {
 private:
  /// Defines the host version of the tensor to be a friend of this class.
  template <typename TT, std::size_t D> friend class HostTensor;

  /// Defines the type of the executor for the iterator to be a GPU executor.
  using exec_t = exec::gpu_type;

 public:
  /// Defines the number of dimensions in the tensor.
  static constexpr auto dimensions = std::size_t{N};

  /// Defines the type of the tensor.
  using self_t            = DeviceTensor;
  /// Defines the type of the elements in the tensor.
  using element_t         = std::decay_t<T>;
  /// Defines an alias for the base tensor class.
  using base_t            = BaseTensor<element_t, N>;
  /// Defines the type of the equivalent host tensor.
  using host_t            = HostTensor<element_t, N>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = typename base_t::pointer_t;
  /// Defines the type of a non const iterator.
  using iterator_t        = StridedIterator<element_t, false, exec_t>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = StridedIterator<element_t, true, exec_t>;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<N>;
  /// Defines the type of a non const iterator.
  using multi_iterator_t  = MultidimIterator<element_t, dim_info_t, exec_t>;

  /// Creates a device tensor with no elements. This requires the tensor to be
  /// resized before using it.
  DeviceTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] dim_sizes The size of each dimension for the tensor..
  /// \tparam    DimSizes  The type of the dimension sizes.
  template <typename... DimSizes>
  DeviceTensor(DimSizes&&... dim_sizes);

  /// Cleans up any memory allocated for the tensor.
  ~DeviceTensor();

  /// Constructor to copy a device tensor to another device tensor.
  /// \param[in] other The other device tensor to copy from.
  DeviceTensor(const self_t& other);

  /// Constructor to move a device tensor to another device tensor.
  /// \param[in] other The other device tensor to move from.
  DeviceTensor(self_t&& other);

  /// Constructor to create a device tensor from a host tensor.
  /// \param[in] host_tensor The host tensor to create the device tensor from.
  DeviceTensor(const host_t& host_tensor);

  /// Overload of operator= to copy a device tensor to another device tensor.
  /// \param[in] other The other device tensor to copy from.
  auto& operator=(const self_t& other);

  /// Overload of operator= to move a device tensor to another device tensor.
  /// \param[in] other The other device tensor to move from.
  auto& operator=(self_t&& other);

  /// Returns the device tensor as a host tensor.
  auto as_host() const;

  /// Returns an iterator to the first element in the tensor.
  fluidity_host_device iterator_t begin()
  {
    return iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  fluidity_host_device iterator_t end()
  {
    return iterator_t{this->_data + this->total_size()};
  }

  /// Returns an iterator to the first element in the tensor.
  /// Calling this with a range-based-for will result in a segfault, 
  /// for example:
  /// 
  /// ~~~{.cpp}
  /// DeviceTensor<int, 1> t(20);
  /// for (const auto& device_element : DeviceTensor<int, 1>(20))
  /// {
  ///   // Do something
  /// }
  /// ~~~
  /// 
  /// This is designed to be used by the algorithms in the fluid:: namespace,
  /// for example:
  /// 
  /// ~~~{.cpp}
  /// auto tensor = DeviceTensor<int, 1>(50);
  /// auto result = fluid::reduce(tensor.begin(), tensor.end(), fluid::sum);
  /// ~~~
  /// 
  /// If you are wanting to use the tensor in a range-based-for loop, then the
  /// tensor needs to be converted to a host version of the tensor, for example:
  /// 
  /// ~~~{.cpp}
  /// auto tensor = DeviceTensor<int, 1>(50);
  /// for (const auto& e : tensor.as_host())
  /// {
  ///   // Do something with element ...
  /// }
  /// ~~~
  fluidity_host_device const_iterator_t begin() const
  {
    return const_iterator_t{this->_data};
  }

  /// Returns an iterator to the last element in the tensor.
  fluidity_host_device const_iterator_t end() const
  {
    return const_iterator_t{this->_data + this->total_size()};
  }

  /// Returns a multi dimensional iterator over the tensor data, which is
  /// initialized to point to the start of the tensor data. For a 1D tensor the
  /// multidimensional iterator behaves the same as a StridedIterator.
  fluidity_host_device multi_iterator_t multi_iterator() const
  {
    return multi_iterator_t{this->_data, this->dim_info()};
  }

  /// Resizes the tensor to contain specific number of elements in each
  /// dimension.
  /// \param[in] dim_sizes The size of each dimension.
  /// \tparam    DimSizes  The type of the dimension sizes.
  template <typename... DimSizes>
  void resize(DimSizes&&... dim_sizes);

  /// Resizes the dimension \p dim to have \p elements number of elements.
  /// \param[in] dim      The dimension to resize.
  /// \param[in] elements The number of elements for the dimension.
  void resize_dim(std::size_t dim, std::size_t elements);

 private:
  bool _must_free = true; //!< Sets if the memory must be freed.

  /// Allocates memory for the array.
  void allocate();

  /// Cleans up the memory allocated for the tensor.
  void cleanup();
};

//==--- DeviceTensor Implementation ----------------------------------------==//

//===== Public ----------------------------------------------------------=====//

template <typename T, std::size_t N> template <typename... DimSizes>
DeviceTensor<T, N>::DeviceTensor(DimSizes&&... dim_sizes)
: base_t{std::forward<DimSizes>(dim_sizes)...}
{
  allocate();
}

template <typename T, std::size_t N>
DeviceTensor<T, N>::~DeviceTensor()
{
  cleanup();
}

template <typename T, std::size_t N>
DeviceTensor<T, N>::DeviceTensor(const self_t& other) : base_t{other}
{
  allocate();
  util::cuda::memcpy_device_to_device(other._data            ,
                                      this->_data            ,
                                      this->mem_requirement());
}

template <typename T, std::size_t N>
DeviceTensor<T, N>::DeviceTensor(self_t&& other) : base_t{other}
{
  this->_data       = other._data;
  this->_must_free  = true;
  other._data       = nullptr;
  other._must_free  = false;
}

template <typename T, std::size_t N>
DeviceTensor<T, N>::DeviceTensor(const host_t& host): base_t{host}
{
  allocate();
  util::cuda::memcpy_host_to_device(host._data             ,
                                    this->_data            ,
                                    this->mem_requirement());
}

template <typename T, std::size_t N>
auto& DeviceTensor<T, N>::operator=(const self_t& other)
{
  this->set_dim_sizes(other);
  this->_must_free = true;
  allocate();
  util::cuda::memcpy_device_to_device(other._data            ,
                                      this->_data            ,
                                      this->mem_requirement());
  return *this;
}

template <typename T, std::size_t N>
auto& DeviceTensor<T, N>::operator=(self_t&& other) 
{
  this->set_dim_sizes(other);
  this->_data      = other._data;
  this->_must_free = true;
  other._data      = nullptr;
  other._must_free = false;
  return *this;
}

template <typename T, std::size_t N>
auto DeviceTensor<T, N>::as_host() const
{
  return host_t(*this);
}

template <typename T, std::size_t N> template <typename... DimSizes>
void DeviceTensor<T, N>::resize(DimSizes&&... dim_sizes)
{
  cleanup();
  this->reset_dim_sizes(std::forward<DimSizes>(dim_sizes)...);
  allocate();
}

template <typename T, std::size_t N>
void DeviceTensor<T, N>::resize_dim(std::size_t dim, std::size_t elements)
{
  cleanup();
  this->_dim_sizes[dim] = elements;
  allocate();
}

//===== Private ---------------------------------------------------------=====//

template <typename T, std::size_t N>
void DeviceTensor<T, N>::allocate()
{
  // TODO: Add an implementation for aligned allocation...
  if (this->_data == nullptr) 
  {
    util::cuda::allocate(reinterpret_cast<void**>(&this->_data),
                         this->mem_requirement());
  }
}

template <typename T, std::size_t N>
void DeviceTensor<T, N>::cleanup()
{
  if (this->_data != nullptr && this->_must_free)
  {
    util::cuda::free(this->_data);
    this->_data = nullptr;
  }
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_DEVICE_TENSOR_HPP