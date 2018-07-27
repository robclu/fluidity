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

/// The BlockSizeImpl struct defines a struct which can be specialized for
/// the different dimension types.
/// \tparam Dimension The dimension to implement to the block size computation
///         for.
template <typename Dimension>
struct BlockSizeImpl;

/// Specialization of the blockcomputation for the x dimension.
template <>
struct BlockSizeImpl<dimx_t> {
  /// Overload of function call operator to return the block size in the
  /// x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockDim.x;
  }
};

/// Specialization of the block size computation for the y dimension.
template <>
struct BlockSizeImpl<dimy_t> {
  /// Overload of function call operator to return the block size in the
  /// y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockDim.y;
  }
};

/// Specialization of the block size computation for the z dimension.
template <>
struct BlockSizeImpl<dimz_t> {
  /// Overload of function call operator to return the block size in the
  /// z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockDim.z;
  }
};


/// The GridSizeImpl struct defines a struct which can be specialized for
/// the different dimension types.
/// \tparam Dimension The dimension to implement to the block size computation
///         for.
template <typename Dimension>
struct GridSizeImpl;

/// Specialization of the grid size computation for the x dimension.
template <>
struct GridSizeImpl<dimx_t> {
  /// Overload of function call operator to return the grid size in the
  /// x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return gridDim.x * BlockSizeImpl<dimx_t>()();
  }
};

/// Specialization of the grid size computation for the y dimension.
template <>
struct GridSizeImpl<dimy_t> {
  /// Overload of function call operator to return the grid size in the
  /// y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return gridDim.y * BlockSizeImpl<dimy_t>()();
  }
};

/// Specialization of the grid size computation for the z dimension.
template <>
struct GridSizeImpl<dimz_t> {
  /// Overload of function call operator to return the block size in the
  /// z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return gridDim.z * BlockSizeImpl<dimz_t>()();
  }
};

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

/// The FlattenedBlockIdImpl struct defines a struct which can be specialized
/// for the different dimension types.
/// \tparam Dimension The dimension to implement to the flattened block index
///         computation for.
template <typename Dimension>
struct FlattenedBlockIdImpl;

/// Specialization of the flattened block index computation for the x dimension.
template <>
struct FlattenedBlockIdImpl<dimx_t> {
  /// Overload of function call operator to return the flattened block index in
  /// the x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * gridDim.y * gridDim.x;
  }
};

/// Specialization of the flattened block index computation for the y dimension.
template <>
struct FlattenedBlockIdImpl<dimy_t> {
  /// Overload of function call operator to return the flattened block index in
  /// the y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.y
         + blockIdx.x * gridDim.y
         + blockIdx.z * gridDim.y * gridDim.x;
  }
};

/// Specialization of the flattened block index computation for the z dimension.
template <>
struct FlattenedBlockIdImpl<dimz_t> {
  /// Overload of function call operator to return the flattened block index in
  /// the z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.z
         + blockIdx.y * gridDim.z
         + blockIdx.x * gridDim.y * gridDim.z;
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

/// The BlockIdImpl struct defines a struct which can be specialized for
/// the different dimension types.
/// \tparam Dimension The dimension to implement to the block index
///         computation for.
template <typename Dimension>
struct BlockIdImpl;

/// Specialization of the block index computation for the x dimension.
template <>
struct BlockIdImpl<dimx_t> {
  /// Overload of function call operator to return the block index in the
  /// x dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.x;
  }
};

/// Specialization of the block index computation for the y dimension.
template <>
struct BlockIdImpl<dimy_t> {
  /// Overload of function call operator to return the block index in the
  /// y dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.y;
  }
};

/// Specialization of the block index computation for the z dimension.
template <>
struct BlockIdImpl<dimz_t> {
  /// Overload of function call operator to return the block index in the
  /// z dimension.
  fluidity_device_only std::size_t operator()() const
  {
    return blockIdx.z;
  }
};

#endif // __CUDACC__  

}} // namespace fluid::detail

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_DETAIL_HPP