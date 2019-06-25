//==--- fluidity/traits/device_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_traits.hpp
/// \brief This file defines traits for devices.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_TRAITS_DEVICE_TRAITS_HPP
#define FLUIDITY_TRAITS_DEVICE_TRAITS_HPP

#include <fluidity/container/tuple.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <type_traits>

namespace fluid  {
namespace traits {
namespace detail {

/// Defines a class which is used to determine if an execution policy is for the
/// CPU.
/// \tparam T The type to determine the execution policy for.
template <typename T>
struct IsCpuPolicy {
  /// Defines if the type T has a CPU execution policy.
  static constexpr auto value = is_same_v<exec::exec_policy_t<T>, exec::cpu_t>;
};

/// Specialization of the CPU policy checking class for the case that there is a
/// list of types for which to determine the execution polic of.
/// \tparam Ts The types to determine the execution policy for.
template <typename... Ts>
struct IsCpuPolicy<Tuple<Ts...>> {
  /// Defines the type of the tuple.
  using tuple_t = Tuple<Ts...>;

  /// Returns true if the execution type of the first element type of the tuple
  /// is for the cpu.
  static constexpr auto value = 
    is_same_v<exec::exec_policy_t<tuple_element_t<0, tuple_t>>, exec::cpu_t>;
};

/// Defines a class which is used to determine if an execution policy is for the
/// GPU.
/// \tparam T The type to determine the execution policy for.
template <typename T>
struct IsGpuPolicy {
  /// Defines if the type T has a GPU execution policy.
  static constexpr auto value = is_same_v<exec::exec_policy_t<T>, exec::gpu_t>;
};

template <typename... Ts>
struct IsGpuPolicy<Tuple<Ts...>> {
  /// Defines the type of the tuple.
  using tuple_t = Tuple<Ts...>;

  /// Returns true if the execution type of the first element type of the tuple
  /// is for the gpu.
  static constexpr auto value = 
    is_same_v<exec::exec_policy_t<tuple_element_t<0, tuple_t>>, exec::gpu_t>;
};

} // namespace detail

/// Returns true if the template parameter type has an execution policy which
/// matches ``cpu_policy`` and specifies that the CPU should be used for
/// execution. If the type T is a Tuple, then this uses the first element of the
/// tuple to check the execution policy.
/// \tparam T The type to check if is a CPU execution policy.
template <typename T>
static constexpr auto is_cpu_policy_v = 
  detail::IsCpuPolicy<std::decay_t<T>>::value;

/// Returns true if the template parameter type has an execution policy which
/// matches ``gpu_policy`` and specifies that the GPU should be used for
/// execution. If the type T is a Tuple, then this uses the first element of the
/// tuple to check the execution policy.
/// \tparam T The type to check if is a GPU execution policy.
template <typename T>
static constexpr auto is_gpu_policy_v = 
  detail::IsGpuPolicy<std::decay_t<T>>::value;

/// Defines a type which is valid if T has an execution policy and it's a CPU
/// execution policy.
/// \tparam T The type to get a CPU execution enabling type for.
template <typename T>
using cpu_enable_t = std::enable_if_t<is_cpu_policy_v<T>, int>;

/// Defines a type which is valid if T has an execution policy and it's a GPU
/// execution policy.
/// \tparam T The type to get a GPU execution enabling type for.
template <typename T>
using gpu_enable_t = std::enable_if_t<is_gpu_policy_v<T>, int>;

}} // namespace fluid::traits

#endif // FLUIDITY_TRAITS_DEVICE_TRAITS_HPP
