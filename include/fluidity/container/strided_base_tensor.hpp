//==--- fluidity/container/strided_base_tensor.hpp --------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  strided_base_tensor.hpp
/// \brief This file defines the functionality which is common to both the host
///        and the device strided tensor implementations.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_STRIDED_BASE_TENSOR_HPP
#define FLUIDITY_CONTAINER_STRIDED_BASE_TENSOR_HPP

#include "tensor_fwrd.hpp"
#include <fluidity/algorithm/accumulate.hpp>
#include <fluidity/dimension/dimension_info.hpp>
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
/// to both host and device strided tensor implementations. This class
/// holds the tensor data and the information about the tensor, but cannot
/// allocate data and set data, which must be done by the derived class. A
/// strided tensor stored the data of the type as SoA, and the type must
/// therefore support that.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
class StridedBaseTensor {
 private:
  /// Allows device tensor to create base tensors.
  template <typename TT, std::size_t D> friend class StridedDeviceTensor;

  /// Make the HostTensor a friend to allow it to create base tensors.
  template <typename TT, std::size_t D> friend class StridedHostTensor;

  /// Defines the type of this tensor.
  using self_t            = StridedBaseTensor;
  /// Defines the type of the data being stored in the tensor.
  using value_t           = std::decay_t<T>;
  /// Defines the primitive type used bt the value type.
  using prim_t            = typename value_t::value_t;
  /// Defines the type of dimension information used for the tensor.
  using dim_info_t        = DimInfo<Dimensions>;
  /// Defines the type of the pointer to the data to store.
  using pointer_t         = prim_t*;
  /// Defines the type of a reference to the data type.
  using reference_t       = value_t&;
  /// Defines the type of a const reference to the data type.
  using const_reference_t = const value_t&;

  /// Defines the number of elements in the underlying type.
  static constexpr auto elements   = value_t::elements;
  /// Defines the number of dimensions for the tensor.
  static constexpr auto dimensions = Dimensions;

  /// Creates a base tensor, with no elements or size. Calling this constructor
  /// required resizing the tensor.
  StridedBaseTensor() = default;

  /// Initializes the size of each of the dimensions in the tensor, and the
  /// total number of elements in the tensor. This is only enabled when the
  /// types of the paramters are integral types.
  /// \param[in] size_0     The size of dimension 0.
  /// \param[in] size_other The sizes of the other dimensions.
  /// \tparam    Size0      The type of the size of dimension 0.
  /// \tparam    SizeOther  The types of the other dimensions.
  template <typename S, typename... Ss, size_enable_t<S> = 0>
  fluidity_host_device StridedBaseTensor(S&& size_0, Ss&&... size_other);

  /// Creates a base tensor from the \p other tensor.
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device StridedBaseTensor(const self_t& other);

  /// Creates a base tensor from the \p other tensor.
  /// \param[in] other The other tensor to create this one from.
  fluidity_host_device self_t& operator=(const self_t& other);

  /// Sets the sizes of the dimensions from another device tensor.
  /// \param[in] other The other tensor to set the sizes from.
  fluidity_host_device void set_dim_sizes(const self_t& other);

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
  fluidity_host_device std::size_t size(std::size_t n) const;

  /// Returns the total number of elements in the tensor.
  fluidity_host_device std::size_t total_size() const;

 protected:
  pointer_t   _data                  = nullptr; //!< Pointer to the data.
  std::size_t _dim_sizes[dimensions] = {};      //!< Elements in each dimension.
};

//==--- StridedBaseTensor Implementation -----------------------------------==//

template <typename T, std::size_t D>
template <typename S, typename... Ss, size_enable_t<S>>
fluidity_host_device
StridedBaseTensor<T, D>::StridedBaseTensor(S&& size_0, Ss&&... size_other)
: _data{nullptr},
  _dim_sizes{static_cast<std::size_t>(size_0),
             static_cast<std::size_t>(size_other)...}
{}

template <typename T, std::size_t D>
fluidity_host_device
StridedBaseTensor<T, D>::StridedBaseTensor(const StridedBaseTensor<T, D>& other)
: _data{nullptr}
{
  set_dim_sizes(other);
}

template <typename T, std::size_t D>
fluidity_host_device StridedBaseTensor<T, D>&
StridedBaseTensor<T, D>::operator=(const StridedBaseTensor<T, D>& other)
{
  _data = nullptr;
  set_dim_sizes(other);
}

template <typename T, std::size_t D>
fluidity_host_device std::size_t
StridedBaseTensor<T, D>::mem_requirement() const
{
  return sizeof(prim_t) * elements * total_size();
}

template <typename T, std::size_t D> template <typename... DimSizes>
fluidity_host_device void
StridedBaseTensor<T, D>::reset_dim_sizes(DimSizes&&... dim_sizes)
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
StridedBaseTensor<T, D>::set_dim_sizes(const StridedBaseTensor<T, D>& other)
{
  unrolled_for<dimensions>([&] (auto i)
  {
    _dim_sizes[i] = other.size(i);
  });
}

template <typename T, std::size_t D>
fluidity_host_device std::size_t
StridedBaseTensor<T, D>::size(std::size_t n) const
{
  return _dim_sizes[n];
}

template <typename T, std::size_t D>
fluidity_host_device std::size_t
StridedBaseTensor<T, D>::total_size() const
{
  std::size_t size = 1;
  unrolled_for<dimensions>([&] (auto i)
  {
    size *= _dim_sizes[i];
  });
  return size;
}

} // namespace fluid

#endif // FLUIDITY_CONTAINER_STRIDED_BASE_TENSOR_HPP