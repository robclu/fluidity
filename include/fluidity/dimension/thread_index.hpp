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

#include "dimension.hpp"

namespace fluid {

#if defined(__CUDACC__)

namespace detail {

/// Implementation for the x dimension case for the thread index.
/// \tparam V   The value which defines the dimension.
template <std::size_t V>
fluidity_device_only std::size_t thread_id_impl(Dimension<V>);

/// Implementation for the x dimension case for the flattened index.
/// \tparam V   The value which defines the dimension.
//template <std::size_t V>
//fluidity_device_only std::size_t flattened_id_impl(Dimension<V>);

template <typename Dimension>
struct FlattenedIdImpl;

template <>
struct FlattenedIdImpl<dimx_t> {
  fluidity_host_device std::size_t operator()() const
  {
    return threadIdx.x + blockIdx.x * blockDim.x;
  }
};

template <>
struct FlattenedIdImpl<dimy_t> {
  fluidity_host_device std::size_t operator()() const
  {
    return threadIdx.y + blockIdx.y * blockDim.y;
  }
};

template <>
struct FlattenedIdImpl<dimz_t> {
  fluidity_host_device std::size_t operator()() const
  {
    return threadIdx.z + blockIdx.z * blockDim.z;
  }
};

} // namespace detail

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t thread_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  return 0; //detail::thread_id_impl(Dimension<Value>{});
}

/// Returns the value of the flattened thread index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only inline std::size_t flattened_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  return detail::FlattenedIdImpl<Dimension<Value>>()();
}

#else

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
fluidity_host_only constexpr inline std::size_t falttened_id(Dimension<Value>)
{
  // \todo, add implementation ... 
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  if constexpr (Value == 0) { return 0; }
  if constexpr (Value == 1) { return 1; }
  if constexpr (Value == 2) { return 2; }
}

#endif 

} // namespace fluid

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_HPP