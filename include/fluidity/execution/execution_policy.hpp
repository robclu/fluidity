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

/// Defines the default number of threads per dimension for 1d.
static constexpr std::size_t default_threads_1d = 512;

/// Defines the default number of threads per dimension for 2d in dim x.
static constexpr std::size_t default_threads_2d_x = 32;
/// Defines the default number of threads per dimension for 2d in dim x.
static constexpr std::size_t default_threads_2d_y = 16;

/// Defines the default number of threads per dimension for 3d in dim x.
static constexpr std::size_t default_threads_3d_x = 16;
/// Defines the default number of threads per dimension for 3d in dim y.
static constexpr std::size_t default_threads_3d_y = 8;
/// Defines the default number of threads per dimension for 3d in dim z.
static constexpr std::size_t default_threads_3d_z = 4;

#if defined(__CUDACC__)

/// Returns the number of threads in each of the dimensions for a single
/// dimension.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_threads_sizes(Iterator&& it)
{
  if (it.num_dimensions() == 1)
  {
    return dim3(it.size(dim_x) < default_threads_1d 
                  ? it.size(dim_x) : default_threads_1d);
  }
  else if (it.num_dimensions() == 2)
  {
    return dim3(it.size(dim_x) < default_threads_2d_x 
                  ? it.size(dim_x) : default_threads_2d_x,
                it.size(dim_y) < default_threads_2d_y
                  ? it.size(dim_y : default_threads_2d_y));
  }
  return dim3(it.size(dim_x) < default_threads_3d_x 
                ? it.size(dim_x) : default_threads_3d_x,
              it.size(dim_y) < default_threads_3d_y
                ? it.size(dim_y) : default_threads_3d_y,
              it.size(dim_z) < default_threads_3d_z
                ? it.size(dim_z) : default_threads_3d_z);
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes)
{
  const auto default_value = unsigned int{1};
  if (it.num_dimensions() == 1)
  {
    return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value));
  }
  else if (it.num_dimensions() == 2)
  {
    return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value),
                std::max(it.size(dim_y) / thread_sizes.y, default_value));
  }
  return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value),
              std::max(it.size(dim_y) / thread_sizes.y, default_value)
              std::max(it.size(dim_z) / thread_sizes.z, default_value));
}

#else

#endif // __CUDACC__

}} // namespace fluid::exec

#endif // FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP