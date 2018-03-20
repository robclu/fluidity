//==--- fluidity/dimension/thread_index_detail.hpp --------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  thread_index_detail.hpp
/// \brief This file contains implementation details of the thread index
///        functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_DIMENSION_THREAD_INDEX_DETAIL_HPP
#define FLUIDITY_DIMENSION_THREAD_INDEX_DETAIL_HPP

#include "dimension.hpp"

namespace fluid  {
namespace detail {

#if defined(__CUDACC__)

/// The FlattenedIdImpl struct defines a struct which can be specialized for
/// the different dimension types.
/// \tparam Dimension The dimension to implement to the flattened index
///         computation for.
template <typename Dimension>
struct FlattenedIdImpl;

/// Specialization of the flattened index computation for the x dimension.
template <>
struct FlattenedIdImpl<dimx_t> {
  /// Overload of function call operator to return the flattened index in the
  /// x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.x + blockIdx.x * blockDim.x;
  }
};

/// Specialization of the flattened index computation for the y dimension.
template <>
struct FlattenedIdImpl<dimy_t> {
  /// Overload of function call operator to return the flattened index in the
  /// y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.y + blockIdx.y * blockDim.y;
  }
};

/// Specialization of the flattened index computation for the z dimension.
template <>
struct FlattenedIdImpl<dimz_t> {
  /// Overload of function call operator to return the flattened index in the
  /// z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.z + blockIdx.z * blockDim.z;
  }
};

/// The ThreadIdImpl struct defines a struct which can be specialized for
/// the different dimension types.
/// \tparam Dimension The dimension to implement to the thread index
///         computation for.
template <typename Dimension>
struct ThreadIdImpl;

/// Specialization of the thread index computation for the x dimension.
template <>
struct ThreadIdImpl<dimx_t> {
  /// Overload of function call operator to return the thread index in the
  /// x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.x;
  }
};

/// Specialization of the thread index computation for the y dimension.
template <>
struct ThreadIdImpl<dimy_t> {
  /// Overload of function call operator to return the thread index in the
  /// y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.y;
  }
};

/// Specialization of the thread index computation for the z dimension.
template <>
struct ThreadIdImpl<dimz_t> {
  /// Overload of function call operator to return the thread index in the
  /// z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return threadIdx.z;
  }
};

#endif // __CUDACC__  

}} // namespace fluid::detail

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_DETAIL_HPP