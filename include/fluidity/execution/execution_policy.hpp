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

#include <fluidity/utility/portability.hpp>
#include <fluidity/dimension/dimension_info.hpp>
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

/// Defines the type of the execution policy of the type T, if it has an
/// execution policy.
/// \tparam T The type to get the execution policy of.
template <typename T>
using exec_policy_t = typename std::decay_t<T>::exec_t;

/// Defines a type which is valid if T has an execution policy and it's a CPU
/// execution policy.
/// \tparam T The type to get a CPU execution enabling type for.
template <typename T>
using cpu_enable_t = std::enable_if_t<is_cpu_policy_v<exec_policy_t<T>>, int>;

/// Defines a type which is valid if T has an execution policy and it's a GPU
/// execution policy.
/// \tparam T The type to get a GPU execution enabling type for.
template <typename T>
using gpu_enable_t = std::enable_if_t<is_gpu_policy_v<exec_policy_t<T>>, int>;

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

// If the compilation system has cuda functionality then set the default
// execution policy to use the GPU.
#if defined(FLUIDITY_CUDA_AVAILABLE)

/// Defines the default type of execution to use.
using default_type = gpu_type;

/// If the compilation system has cuda functionality then set the default
/// execution policy to use the GPU.
static constexpr auto default_policy = gpu_policy;

#else

/// Defines the default type of execution to use.
using default_type = cpu_type;

/// If the compilation system has no cuda functionality then set the default
/// execution policy to use the CPU.
static constexpr auto default_policy = cpu_policy;

#endif // FLUIDITY_CUDA_AVAILABLE

#if defined(__CUDACC__)

namespace detail {

/// Returns the number of threads in each of the dimensions for a single
/// dimension. This overload is selected for a 1D iterator.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, dispatch_tag_1d_t)
{
  return dim3(it.size(dim_x) < default_threads_1d 
                ? it.size(dim_x) : default_threads_1d);
}

/// Returns the number of threads in each of the dimensions for a two
/// dimensional iterator. This overload is selected for a 2D iterator.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, dispatch_tag_2d_t)
{
  return dim3(it.size(dim_x) < default_threads_2d_x 
                ? it.size(dim_x) : default_threads_2d_x,
              it.size(dim_y) < default_threads_2d_y
                ? it.size(dim_y) : default_threads_2d_y);
}

/// Returns the number of threads in each of the dimensions for a three
/// dimensional iterator. This overload is selected for a 3D iterator.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, dispatch_tag_3d_t)
{
  return dim3(it.size(dim_x) < default_threads_3d_x 
                ? it.size(dim_x) : default_threads_3d_x,
              it.size(dim_y) < default_threads_3d_y
                ? it.size(dim_y) : default_threads_3d_y,
              it.size(dim_z) < default_threads_3d_z
                ? it.size(dim_z) : default_threads_3d_z);
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 1D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes, dispatch_tag_1d_t)
{
  constexpr auto default_value = std::size_t{1};
  return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value));
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 2D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes, dispatch_tag_2d_t)
{
  constexpr auto default_value = std::size_t{1};
  return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value),
              std::max(it.size(dim_y) / thread_sizes.y, default_value));
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 3D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes, dispatch_tag_3d_t)
{
  constexpr auto default_value = std::size_t{1};
  return dim3(std::max(it.size(dim_x) / thread_sizes.x, default_value),
              std::max(it.size(dim_y) / thread_sizes.y, default_value),
              std::max(it.size(dim_z) / thread_sizes.z, default_value));
}

} // namespace detail

/// Returns the number of threads for each dimension.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it)
{
  return detail::get_thread_sizes(std::forward<Iterator>(it),
                                  dim_dispatch_tag<Iterator>);
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes)
{
  return detail::get_block_sizes(std::forward<Iterator>(it),
                                 thread_sizes              ,
                                 dim_dispatch_tag<Iterator>);
}

#else

#endif // __CUDACC__

}} // namespace fluid::exec

#endif // FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP