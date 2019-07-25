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

#include "tensor_fwrd.hpp"
#include <fluidity/algorithm/accumulate.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/traits/container_traits.hpp>
#include <fluidity/utility/portability.hpp>
#include <cstddef>

namespace fluid {

/// Defines a type if the type T is integral, which can be used to enable a
/// parameter pack if the pack types are integral.
/// \tparam T The type to check if integral.
template <typename T>
using size_enable_t = 
  std::enable_if_t<std::is_integral<std::decay_t<T>>::value, int>;

/// Implementation of a base tensor class which contains functionality common
/// to both host and device implementations of the tensor class. More
/// specifically, this class holds the tensor data and metadata for the tensor,
/// but cannot allocate data and set data, since this is an implementation
/// detail of the implementation (for example, device tensors need to allocate
/// the memory on the device).
/// \tparam T The type of the data to store in the tensor.
/// \tparam D The number of dimensions for the tensor.
template <typename T, std::size_t D>
class BaseTensor {
 private:
  /// Allows a device tensor to create base tensors.
  template <typename TT, std::size_t DD> friend class DeviceTensor;
  /// Allows a host tensor to create base tensors.
  template <typename TT, std::size_t DD> friend class HostTensor;

  /// Defines the type of the data being stored in the tensor.
  using value_t           = std::decay_t<T>;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<D>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = value_t*;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&;

  /// Defines the number of dimensions for the tensor.
  static constexpr auto dimensions = D;

  /// Creates a base tensor, with no elements or size. Calling this constructor
  /// required resizing the tensor.
  BaseTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor.
  ///
  /// This is only enabled when the type of template parameters are not
  /// containers.
  ///
  /// \param[in] dim_sizes The size of each dimension for the tensor..
  /// \tparam    DimSizes  The type of the dimension sizes.
  template <typename S1, typename... Sz, size_enable_t<S1> = 0>
  fluidity_host_device BaseTensor(S1&& size_one, Sz&&... other_sizes)
  : _data{nullptr},
    _dim_sizes{
      static_cast<std::size_t>(size_one),
      static_cast<std::size_t>(other_sizes)...
    } {}

  /// Creates a base tensor from the \p other tensor.
  /// This only sets the sizes of this base tensor from the other base tensor.
  /// It does not set the data from the tensor as the data to set depends on
  /// the implementation.
  ///
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device BaseTensor(const BaseTensor& other)
  : _data{nullptr} {
    set_dim_sizes(other);
  }

  /// Creates a base tensor from the \p other tensor, copying the metadata from
  /// the \p other tensor.
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device BaseTensor& operator=(const BaseTensor& other) {
    _data = nullptr;
    set_dim_sizes(other);
  }

  /// Sets the sizes of the dimensions from another device tensor.
  /// \param[in] other The other tensor to set the sizes from.
  fluidity_host_device void set_dim_sizes(const BaseTensor& other);

  /// Sets/resets the sizes of the dimensions.
  /// \param[in] dim_sizes The sizes of the dimensions.
  /// \tparam    DimSizes  The type of the dimension sizes.
  template <typename... DimSizes>
  fluidity_host_device void reset_dim_sizes(DimSizes&&... dim_sizes);

 public:
  /// Returns a reference to the element at position i in the tensor. This
  /// is independent of the dimensionality of the tensor, it treats the tensor
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

  /// Returns the dimension information for the tensor.
  fluidity_host_device auto dim_info() const
  {
    dim_info_t info;
    unrolled_for<dimensions>([&] (auto i)
    {
      info[i] = this->_dim_sizes[i];
    });
    return info;
  }

  /// Returns the data for the tensor.
  fluidity_host_device void reset_data(pointer_t new_data)
  {
    _data = new_data;
  }

  /// Returns the amount of memory required by the tensor, in bytes.
  fluidity_host_device std::size_t mem_requirement() const;

  /// Returns the size of the \p nth dimension.
  /// \param[in] n The index of the dimension to get the size of.
  fluidity_host_device auto size(std::size_t n) const -> std::size_t {
    return _dim_sizes[n];
  }

  /// Returns the total number of elements in the tensor.
  fluidity_host_device auto size() const -> std::size_t {
    std::size_t sz = 1;
    unrolled_for<dimensions>([&] (auto i) {
      sz *= _dim_sizes[i];
    });
    return sz;
  }

  /// Returns the total number of elements in the tensor.
  /// TODO: Deprecate!
  fluidity_host_device std::size_t total_size() const;

 protected:
  pointer_t   _data                  = nullptr; //!< Pointer to the data.
  std::size_t _dim_sizes[dimensions] = {};      //!< Elements in each dimension.
};

//==--- BaseTensor Implementation -----------------------------------------==//

template <typename T, std::size_t D>
fluidity_host_device std::size_t BaseTensor<T, D>::mem_requirement() const
{
  return sizeof(value_t) * total_size();
}

template <typename T, std::size_t D> template <typename... DimSizes>
fluidity_host_device void
BaseTensor<T, D>::reset_dim_sizes(DimSizes&&... dim_sizes)
{
  static_assert(sizeof...(DimSizes) == dimensions,
                "Must specify size for each dimension when resizing.");
  std::size_t sizes[dimensions] = { static_cast<std::size_t>(dim_sizes)... };
  unrolled_for<dimensions>([&] (auto i)
  {
    _dim_sizes[i] = sizes[i];
  });
}

template <typename T, std::size_t D>
fluidity_host_device void
BaseTensor<T, D>::set_dim_sizes(const BaseTensor<T, D>& other)
{
  unrolled_for<dimensions>([&] (auto i)
  {
    _dim_sizes[i] = other.size(i);
  });
}

template <typename T, std::size_t D>
fluidity_host_device std::size_t BaseTensor<T, D>::total_size() const
{
  std::size_t size = 1;
  unrolled_for<dimensions>([&] (auto i)
  {
    size *= _dim_sizes[i];
  });
  return size;
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_BASE_TENSOR_HPP
