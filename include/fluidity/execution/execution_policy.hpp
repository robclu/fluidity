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

#include <fluidity/container/tuple.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/utility/portability.hpp>
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

/// Defines the type of the execution policy of the type T, if it has an
/// execution policy.
/// \tparam T The type to get the execution policy of.
template <typename T>
using exec_policy_t = typename std::decay_t<T>::exec_t;

namespace detail {

/// Defines a class which is used to determine if an execution policy is for the
/// CPU.
/// \tparam T The type to determine the execution policy for.
template <typename T>
struct IsCpuPolicy {
  /// Defines if the type T has a CPU execution policy.
  static constexpr auto value = is_same_v<exec_policy_t<T>, cpu_type>;
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
    is_same_v<exec_policy_t<tuple_element_t<0, tuple_t>>, cpu_type>;
};

/// Defines a class which is used to determine if an execution policy is for the
/// GPU.
/// \tparam T The type to determine the execution policy for.
template <typename T>
struct IsGpuPolicy {
  /// Defines if the type T has a GPU execution policy.
  static constexpr auto value = is_same_v<exec_policy_t<T>, gpu_type>;
};

template <typename... Ts>
struct IsGpuPolicy<Tuple<Ts...>> {
  /// Defines the type of the tuple.
  using tuple_t = Tuple<Ts...>;

  /// Returns true if the execution type of the first element type of the tuple
  /// is for the gpu.
  static constexpr auto value = 
    is_same_v<exec_policy_t<tuple_element_t<0, tuple_t>>, gpu_type>;
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
dim3 get_thread_sizes(Iterator&& it, tag_1d_t, std::size_t pad = 0)
{
  const auto elements = it.size(dim_x) - pad;
  return dim3(elements < default_threads_1d ? elements : default_threads_1d);
}

/// Returns the number of threads in each of the dimensions for a two
/// dimensional iterator. This overload is selected for a 2D iterator.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, tag_2d_t, std::size_t pad = 0)
{
  const auto elements_x = it.size(dim_x) - pad;
  const auto elements_y = it.size(dim_y) - pad;
  return dim3(elements_x < default_threads_2d_x 
               ? elements_x : default_threads_2d_x,
              elements_y < default_threads_2d_y
               ? elements_y : default_threads_2d_y);
}

/// Returns the number of threads in each of the dimensions for a three
/// dimensional iterator. This overload is selected for a 3D iterator.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, tag_3d_t, std::size_t pad = 0)
{
  const auto elements_x = it.size(dim_x) - pad;
  const auto elements_y = it.size(dim_y) - pad;
  const auto elements_z = it.size(dim_z) - pad;
  return dim3(elements_x < default_threads_3d_x 
                ? elements_x : default_threads_3d_x,
              elements_y < default_threads_3d_y
                ? elements_y : default_threads_3d_y,
              elements_z < default_threads_3d_z
                ? elements_z : default_threads_3d_z);
}

/// Returns the number of blocks required based on the number of \p cells and
/// the number of \p threads available. This does all the necessary type
/// conversions.
/// \param[in] cells    The number of cells in the domain.
/// \param[in] threads  The number of threads available.
/// \tparam    Cells    The type of the cell variable.
/// \tparam    Threads  The type of the threads variable.
template <typename Cells, typename Threads>
auto get_num_blocks(Cells cells, Threads threads)
{
   return static_cast<std::size_t>(
     std::max(
       static_cast<std::size_t>(
         std::ceil(static_cast<double>(cells) / static_cast<double>(threads))),
       std::size_t{1}
     )
   );  
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 1D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it          ,
                     dim3       thread_sizes,
                     tag_1d_t               ,
                     std::size_t pad = 0    )
{
  return dim3(get_num_blocks(it.size(dim_x) - pad, thread_sizes.x));
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 2D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it          ,
                     dim3       thread_sizes,
                     tag_2d_t               ,
                     std::size_t pad = 0    )
{
  return dim3(get_num_blocks(it.size(dim_x) - pad, thread_sizes.x),
              get_num_blocks(it.size(dim_y) - pad, thread_sizes.y));
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes. This overload is for a 3D space.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it          ,
                     dim3       thread_sizes,
                     tag_3d_t               ,
                     std::size_t pad = 0    )
{
  return dim3(get_num_blocks(it.size(dim_x) - pad, thread_sizes.x),
              get_num_blocks(it.size(dim_y) - pad, thread_sizes.y),
              get_num_blocks(it.size(dim_z) - pad, thread_sizes.z));
}

} // namespace detail

/// Returns the number of threads for each dimension.
/// \param[in] it       The iterator for the multi dimensional space.
/// \tparam    Iterator The type of the iterator. 
template <typename Iterator>
dim3 get_thread_sizes(Iterator&& it, std::size_t pad = 0)
{
  using iter_t = std::decay_t<Iterator>;
  return detail::get_thread_sizes(std::forward<Iterator>(it)          ,
                                  dim_dispatch_tag<iter_t::dimensions>,
                                  pad                                 );
}

/// Returns the size of the block based on the size of the space defined by the
/// \p iterator and the thread sizes.
/// \param[in] it           The iterator for the multi dimensional space.
/// \param[in] thread_sizes The number of threads in each dimension.
/// \tparam    Iterator     The type of the iterator. 
template <typename Iterator>
dim3 get_block_sizes(Iterator&& it, dim3 thread_sizes, std::size_t pad = 0)
{
  using iter_t = std::decay_t<Iterator>;
  return detail::get_block_sizes(std::forward<Iterator>(it)          ,
                                 thread_sizes                        ,
                                 dim_dispatch_tag<iter_t::dimensions>,
                                 pad                                 );
}

#else

#endif // __CUDACC__

}} // namespace fluid::exec

#endif // FLUIDITY_EXECUTION_EXECUTION_POLICY_HPP
