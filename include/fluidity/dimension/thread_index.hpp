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

/// Returns the value of the thread index in a given dimension. The dimension
/// must be one of dim_x, dim_y, dim_z, or else a compile time error will be
/// generated.
/// \param[in] dim    The dimension to get the thread index for.
/// \tparam    Value  The value which defines the dimension.
template <std::size_t Value>
fluidity_device_only constexpr inline std::size_t thread_id(Dimension<Value>)
{
  static_assert(Value <= 2, "Can only get thread id for 3 dimensions {0,1,2}.");
  if constexpr (Value == 0) { return threadIdx.x; }
  if constexpr (Value == 1) { return threadIdx.y; }
  if constexpr (Value == 2) { return threadIdx.z; }
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