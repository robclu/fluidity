//==--- fluidity/traits/tensor_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_traits.hpp
/// \brief This file defines traits related to containers.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_TRAITS_TENSOR_TRAITS_HPP
#define FLUIDITY_TRAITS_TENSOR_TRAITS_HPP

#include <fluidity/container/tensor_fwrd.hpp>

namespace fluid  {
namespace traits {
namespace detail {

/// Struct which is specialized for device tensors.
/// \tparam T The type to determine if is a device tensor.
template <typename T>
struct IsDeviceTensor {
  /// Defines that the type is not a device tensor.
  static constexpr auto value = false;
};

/// Specialization for device tensors.
/// \tparam T     The type of the tensor data
/// \tparam Dims  The number of dimensions for the tensor.
template <typename T, std::size_t Dims>
struct IsDeviceTensor<DeviceTensor<T, Dims>> {
  /// Defines that the class is a device tensor.
  static constexpr auto value = true;
};

/// Struct which is specialized for host tensors.
/// \tparam T The type to determine if is a host tensor.
template <typename T>
struct IsHostTensor {
  /// Defines that the type is not a host tensor.
  static constexpr auto value = false;
};

/// Specialization for host tensors.
/// \tparam T     The type of the tensor data
/// \tparam Dims  The number of dimensions for the tensor.
template <typename T, std::size_t Dims>
struct IsHostTensor<HostTensor<T, Dims>> {
  /// Defines that the class is a host tensor.
  static constexpr auto value = true;
};

} // namespace detail

/// Returns true if the type T is a DeviceTensor, otherwise returns false.
/// \tparam T THe type to determine if is a device tensor.
template <typename T>
static constexpr auto is_dtensor_v = 
  detail::IsDeviceTensor<std::decay_t<T>>::value;

/// Returns true if the type T is a DeviceTensor, otherwise returns false.
/// \tparam T THe type to determine if is a device tensor.
template <typename T>
static constexpr auto is_htensor_v = 
  detail::IsHostTensor<std::decay_t<T>>::value;

/// Returns true if the type T is either a host or device tensor, otherwise
/// returns false.
/// \tparam T The type to determine if is a tensor.
template <typename T>
static constexpr auto is_tensor_v = is_dtensor_v<T> || is_htensor_v<T>;

/// Defines a valid type if the type T is a device tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using dtensor_enable_t = std::enable_if_t<is_dtensor_v<T>, int>;

/// Defines a valid type if the type T is a not device tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using non_dtensor_enable_t = std::enable_if_t<!is_dtensor_v<T>, int>;

/// Defines a valid type if the type T is a host tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using htensor_enable_t = std::enable_if_t<is_htensor_v<T>, int>;

/// Defines a valid type if the type T is a not device tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using non_htensor_enable_t = std::enable_if_t<!is_htensor_v<T>, int>;

/// Defines a valid type if the type T is either a host or device tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using tensor_enable_t = std::enable_if_t<is_tensor_v<T>, int>;

/// Defines a valid type if the type T is a not device tensor or host tensor.
/// \tparam T The type to base the enabling on.
template <typename T>
using non_tensor_enable_t = std::enable_if_t<!is_tensor_v<T>, int>;

}} // namespace fluid::traits

#endif // FLUIDITY_TRAITS_TENSOR_TRAITS_HPP
