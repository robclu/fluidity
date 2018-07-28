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
template <std::size_t Value>
fluidity_device_only inline std::size_t block_size(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get size for 3 dimensions {0,1,2}.");
  return detail::BlockSizeImpl<Dimension<Value>>()();
}

/// Returns the size of the block in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t grid_size(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get size for 3 dimensions {0,1,2}.");
  return detail::GridSizeImpl<Dimension<Value>>()();
}

/// Returns the total size of the grid (total number of threads in the grid).
fluidity_device_only inline std::size_t grid_size()
{
  return grid_size(dim_x) * grid_size(dim_y) * grid_size(dim_z);
}

/// Returns the value of the flattened thread index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated. This returns the global flattened index.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t flattened_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  return detail::FlattenedIdImpl<Dimension<Value>>()();
}

/// Returns the value of the flattened block index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t flattened_block_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  return detail::FlattenedBlockIdImpl<Dimension<Value>>()();
}

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t thread_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  return detail::ThreadIdImpl<Dimension<Value>>()();
}

/// Returns the value of the block index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the block index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t block_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get block id for 3 dimensions {0,1,2}.");
  return detail::BlockIdImpl<Dimension<Value>>()();
}

#else // __CUDACC__

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_host_only constexpr inline std::size_t thread_id(Dimension<Value>)
{
  // \todo, add implementation ... 
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  if constexpr (Value == 0) { return 0; }
  if constexpr (Value == 1) { return 1; }
  if constexpr (Value == 2) { return 2; }
}

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_host_only constexpr inline std::size_t flattened_id(Dimension<Value>)
{
  // \todo, add implementation ... 
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  if constexpr (Value == 0) { return 0; }
  if constexpr (Value == 1) { return 1; }
  if constexpr (Value == 2) { return 2; }
}

#endif // __CUDACC__

/// Returns true if the global thread index in each dimension is less than the
/// size of the iterator in the dimension.
/// \param[in]  it The iterator over the space to determine if in range.
/// \tparam     It The type of the iterator.
template <typename It>
fluidity_host_device bool in_range(It&& it)
{
  using iter_t = std::decay_t<It>;
  bool result = true;
  unrolled_for<iter_t::dimensions>([&] (auto i)
  {
    constexpr auto dim = Dimension<i>();
    result *= flattened_id(dim) < it.size(dim);
  });
  return result;
}

} // namespace fluid

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_HPP