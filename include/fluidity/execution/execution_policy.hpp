//==--- fluidity/execution/execution_policy.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_policy.hpp
/// \brief This file defines the options for execution policy for executing
///        algorithms.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP
#define FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP

#include <fluidity/utility/type_traits.hpp>

namespace fluid {
namespace exec  {

/// The DeviceKind enumeration defines the type of device used for exection.
enum class DeviceKind {
  cpu = 0,  //!< Defines execution on the CPU.
  gpu = 1   //!< Defines execution on the GPU.
};

/// Defines a policy class for the execution which contains information
/// for the execution.
/// \tparam Device  The kind of the execution device for the policy.
template <DeviceKind Device>
struct ExecutionPolicy {
  /// Defines the type of device used for the execution.
  static constexpr auto device = Device;
};

/// Defines an alias for an execution policy type which uses a cpu.
using cpu_type = ExecutionPolicy<DeviceKind::cpu>;

/// Defines an alias for an execution policy type which uses a gpu.
using gpu_type = ExecutionPolicy<DeviceKind::gpu>;

/// Defines an execution policy instance which uses a cpu.
static constexpr auto cpu_policy = cpu_type{};

/// Defines an execution policy instance which uses a gpu.
static constexpr auto gpu_policy = gpu_type{};

/// Returns true if the template parameter type is a ``cpu_policy`` and
/// specifies that the CPU should be used for execution.
/// \tparam T The type to check if is a CPU execution policy.
template <typename T>
static constexpr auto is_cpu_policy_v = 
  is_same_v<std::decay_t<T>, cpu_type>;

/// Returns true if the template parameter type is a ``gpu_policy`` and
/// specifies that the GPU should be used for execution.
/// \tparam T The type to check if is a GPU execution policy.
template <typename T>
static constexpr auto is_gpu_policy_v = 
  is_same_v<std::decay_t<T>, gpu_type>;

}} // namespace fluid::exec

#endif // FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP