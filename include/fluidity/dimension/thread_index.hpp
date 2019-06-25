//==--- fluidity/dimension/thread_index.hpp ---------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  thread_index.hpp
/// \brief This file defines utility functions to access thread indices.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_DIMENSION_THREAD_INDEX_HPP
#define FLUIDITY_DIMENSION_THREAD_INDEX_HPP

#include "thread_index_detail.hpp"
#include <fluidity/algorithm/unrolled_for.hpp>

namespace fluid {

#if defined(__CUDACC__)

/// Returns the total size of the block (total number of threads in the block).
fluidity_device_only inline std::size_t block_size()
{
  return blockDim.x * blockDim.y * blockDim.z;
}

/// Returns the size of the block in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t block_size(Dim dim)
{
  return detail::block_size_impl(dim);
}

/// Returns the size of the block in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t grid_size(Dim dim)
{
  return detail::grid_size_impl(dim);
}

/// Returns the total size of the grid (total number of threads in the grid).
fluidity_device_only inline std::size_t grid_size()
{
  return grid_size(dim_x) * grid_size(dim_y) * grid_size(dim_z);
}

/// Returns the value of the flattened thread index
fluidity_device_only inline std::size_t flattened_thread_id() {
  return threadIdx.x + 
         threadIdx.y * blockDim.x +
         threadIdx.z * blockDim.x * blockDim.y;
}

/// Returns the value of the flattened thread index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated. This returns the global flattened index.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t flattened_id(Dim dim)
{
  return detail::flattened_id_impl(dim);
}

/// Returns the value of the flattened block index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t flattened_block_id(Dim dim)
{
  return detail::flattened_block_id_impl(dim);
}

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t thread_id(Dim dim)
{
  return detail::thread_id_impl(dim);
}

/// Returns the value of the block index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the block index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_device_only inline std::size_t block_id(Dim dim)
{
  return detail::block_id_impl(dim);
}

#else // __CUDACC__

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_host_only constexpr std::size_t thread_id(Dim&& dim)
{
  // \todo, add implementation ... 
  return 0;
}

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <typename Dim>
fluidity_host_only constexpr std::size_t flattened_id(Dim&& dim)
{
  // \todo, add implementation ... 
  return 0;
}

#endif // __CUDACC__

/// Returns true if the global thread index in each dimension is less than the
/// size of the iterator in the dimension.
/// \param[in]  it The iterator over the space to determine if in range.
/// \tparam     It The type of the iterator.
template <typename It>
fluidity_device_only bool in_range(It&& it, std::size_t padding = 0)
{
  using iter_t = std::decay_t<It>;
  bool result  = true;
  unrolled_for<iter_t::dimensions>([&] (auto i)
  {
    result *= flattened_id(i) < it.size(i) - padding;
  });
  return result;
}

} // namespace fluid

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_HPP
