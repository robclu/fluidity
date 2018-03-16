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
template <std::size_t V>
fluidity_device_only std::size_t thread_id_impl(Dimension<V>);

/*
/// Implementation for the x dimension case for the thread index.
fluidity_device_only std::size_t thread_id(dim_y);

/// Implementation for the x dimension case for the thread index.
fluidity_device_only std::size_t thread_id(dim_z);
*/

/// Implementation for the x dimension case for the flattened index.
template <std::size_t V>
fluidity_device_only std::size_t flattened_id_impl(Dimension<V>);

/*
/// Implementation for the y dimension case for the flattened index.
fluidity_device_only std::size_t flattened_id_impl(dim_y);

/// Implementation for the z dimension case for the flattened index.
fluidity_device_only std::size_t flattened_id_impl(dim_z);
*/
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
  return detail::thread_id_impl(Dimension<Value>{});
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
  return detail::flattened_id_impl(Dimension<Value>{});
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

#endif 

} // namespace fluid

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_HPP