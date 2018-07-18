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
#include <fluidity/utility/portability.hpp>
#include <cstddef>
#include <type_traits>

namespace fluid {

/// Defines a type if the type T is integral, which can be used to enable a
/// parameter pack if the pack types are integral.
/// \tparam T The type to check if integral.
template <typename T>
using size_enable_t = 
  std::enable_if_t<std::is_integral<std::decay_t<T>>::value, int>;

/// Implementation of a base tensor class which contains functionality common
/// to both host and device implementations of the tensor class. This class
/// holds the tensor data and the information about the tensor, but cannot
/// allocate data and set data, which must be done by the derived class.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
class BaseTensor {
 private:
  /// Allows device tensor to create base tensors.
  template <typename TT, std::size_t D> friend class DeviceTensor;

  /// Make the HostTensor a friend to allow it to create base tensors.
  template <typename TT, std::size_t D> friend class HostTensor;

  /// Defines the type of the data being stored in the tensor.
  using value_t           = std::decay_t<T>;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<Dimensions>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = value_t*;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&;

  /// Defines the number of dimensions for the tensor.
  static constexpr auto dimensions = Dimensions;

  /// Creates a base tensor, with no elements or size. Calling this constructor
  /// required resizing the tensor.
  BaseTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor. This is only enabled when the
  /// types of the paramters are integral types.
  /// \param[in] size_0     The size of dimension 0.
  /// \param[in] size_other The sizes of the other dimensions.
  /// \tparam    Size0      The type of the size of dimension 0.
  /// \tparam    SizeOther  The types of the other dimensions.
  template <typename Size0, typename... SizeOther, size_enable_t<Size0> = 0>
  fluidity_host_device BaseTensor(Size0&& size_0, SizeOther&&... size_other);

  /// Creates a base tensor from the \p other tensor.
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device BaseTensor(const BaseTensor& other);

  /// Creates a base tensor from the \p other tensor.
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device BaseTensor& operator=(const BaseTensor& other);

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

  /// Returns the amount of memory required by the tensor, in bytes.
  fluidity_host_device std::size_t mem_requirement() const;

  /// Returns the size of the \p nth dimension.
  /// \param[in] n The index of the dimension to get the size of.
  fluidity_host_device std::size_t size(std::size_t n) const;

  /// Returns the total number of elements in the tensor.
  fluidity_host_device std::size_t total_size() const;

 protected:
  pointer_t   _data                  = nullptr; //!< Pointer to the data.
  std::size_t _dim_sizes[dimensions] = {};      //!< Elements in each dimension.
};

//==--- BaseTensor Implementation -----------------------------------------==//

template <typename T, std::size_t D>
template <typename Size0, typename... SizeOther, size_enable_t<Size0>>
fluidity_host_device
BaseTensor<T, D>::BaseTensor(Size0&& size_0, SizeOther&&... size_other)
: _data{nullptr},
  _dim_sizes{static_cast<std::size_t>(size_0),
             static_cast<std::size_t>(size_other)...}
{}

template <typename T, std::size_t D>
fluidity_host_device BaseTensor<T, D>::BaseTensor(const BaseTensor<T, D>& other)
: _data{nullptr}
{
  set_dim_sizes(other);
}

template <typename T, std::size_t D>
fluidity_host_device BaseTensor<T, D>&
BaseTensor<T, D>::operator=(const BaseTensor<T, D>& other)
{
  _data = nullptr;
  set_dim_sizes(other);
}

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
fluidity_host_device std::size_t BaseTensor<T, D>::size(std::size_t n) const
{
  return _dim_sizes[n];
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