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
#include "host_tensor.hpp"
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/iterator/strided_iterator.hpp>
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
  using element_t         = std::decay_t<T>;
  /// Defines an alias for the base tensor class.
  using base_t            = BaseTensor<element_t, 1>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = typename base_t::pointer_t;
  /// Defines the type of a non const iterator.
  using iterator_t        = StridedIterator<self_t, false, exec_t>;
  /// Defines the type of a const iterator.
  using const_iterator_t  = StridedIterator<self_t, true, exec_t>;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<1>;
  /// Defines the type of a non const iterator.
  using multi_iterator_t  = MultidimIterator<element_t, dim_info_t, exec_t>;

  /// Creates a device tensor with no elements. This requires the tensor to be
  /// resized before using it.
  DeviceTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  /// \param[in] elements The number of elements in the 1D ensor.
  DeviceTensor(std::size_t elements);

  /// Cleans up any memory allocated for the tensor.
  ~DeviceTensor();

  /// Constructor to create a device tensor from a host tensor.
  /// \param[in] host_tensor The host tensor to create the device tensor from.
  DeviceTensor(const HostTensor<T, 1>& host_tensor);

  /// Returns the device tensor as a host tensor.
  HostTensor<T, 1> as_host() const;

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
    return const_iterator_t{this->_data + this->_size};
  }

  /// Returns a multi dimensional iterator over the tensor data, which is
  /// initialized to point to the start of the tensor data. For a 1D tensor the
  /// multidimensional iterator behaves the same as a StridedIterator.
  fluidity_host_device multi_iterator_t multi_iterator() const
  {
    return multi_iterator_t{this->data, this->_size};
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
HostTensor<T, 1> DeviceTensor<T, 1>::as_host() const
{
  return HostTensor<T, 1>(*this);
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