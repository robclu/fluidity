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

/// Returns the block size in the x dimension.
fluidity_device_only inline std::size_t block_size_impl(dimx_t)
{
  return blockDim.x;
}

/// Returns the block size in the y dimension.
fluidity_device_only inline std::size_t block_size_impl(dimy_t)
{
  return blockDim.y;
}

/// Returns the block size in the z dimension.
fluidity_device_only inline std::size_t block_size_impl(dimz_t)
{
  return blockDim.z;
}

/// Returns the block size in the \p dim dimension.
/// \param[in] dim The dimension to get the block size for.
fluidity_device_only inline std::size_t block_size_impl(std::size_t dim)
{
  return dim == dimx_t::value ? block_size_impl(dim_x) :
         dim == dimy_t::value ? block_size_impl(dim_y) :
         dim == dimz_t::value ? block_size_impl(dim_z) : 0;
}

/// Returns the grid size in the x dimension.
fluidity_device_only inline std::size_t grid_size_impl(dimx_t)
{
  return gridDim.x * block_size_impl(dim_x);
}

/// Returns the grid size in the y dimension.
fluidity_device_only inline std::size_t grid_size_impl(dimy_t)
{
  return blockDim.y * block_size_impl(dim_y);
}

/// Returns the grid size in the z dimension.
fluidity_device_only inline std::size_t grid_size_impl(dimz_t)
{
  return blockDim.z * block_size_impl(dim_z);
}

/// Returns the grid size in the \p dim dimension.
/// \param[in] dim The dimension to get the grid size for.
fluidity_device_only inline std::size_t grid_size_impl(std::size_t dim)
{
  return dim == dimx_t::value ? grid_size_impl(dim_x) :
         dim == dimy_t::value ? grid_size_impl(dim_y) :
         dim == dimz_t::value ? grid_size_impl(dim_z) : 0;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the x direction.
fluidity_device_only inline std::size_t flattened_id_impl(dimx_t)
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the y direction.
fluidity_device_only inline std::size_t flattened_id_impl(dimy_t)
{
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the z direction.
fluidity_device_only inline std::size_t flattened_id_impl(dimz_t)
{
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the flattened index in the \p dim dimension.
/// \param[in] dim The dimension to get the flattened index for.
fluidity_device_only inline std::size_t flattened_id_impl(std::size_t dim)
{
  return dim == dimx_t::value ? flattened_id_impl(dim_x) :
         dim == dimy_t::value ? flattened_id_impl(dim_y) :
         dim == dimz_t::value ? flattened_id_impl(dim_z) : 0;
}

/// Returns the block index if the blocks were flattened in the x direction.
fluidity_device_only inline std::size_t flattened_block_id_impl(dimx_t)
{
  return blockIdx.x             +
         blockIdx.y * gridDim.x +
         blockIdx.z * gridDim.y * gridDim.x;
}


/// Returns the block index if the blocks were flattened in the y direction.
fluidity_device_only inline std::size_t flattened_block_id_impl(dimy_t)
{
  return blockIdx.y             +
         blockIdx.x * gridDim.y +
         blockIdx.z * gridDim.y * gridDim.x;
}

/// Returns the block index if the blocks were flattened in the z direction.
fluidity_device_only inline std::size_t flattend_block_id_impl(dimz_t)
{
  return blockIdx.z             +
         blockIdx.y * gridDim.z +
         blockIdx.x * gridDim.y * gridDim.z;
}

/// Returns the flattened block index in the \p dim dimension.
/// \param[in] dim The dimension to get the flattened block index for.
fluidity_device_only inline std::size_t flattened_block_id_impl(std::size_t dim)
{
  return dim == dimx_t::value ? flattened_block_id_impl(dim_x) :
         dim == dimy_t::value ? flattened_block_id_impl(dim_y) :
         dim == dimz_t::value ? flattened_block_id_impl(dim_z) : 0;
}

/// Returns the thread index in the x direction.
fluidity_device_only inline std::size_t thread_id_impl(dimx_t)
{
  return threadIdx.x;
}

/// Returns the thread index in the y direction.
fluidity_device_only inline std::size_t thread_id_impl(dimy_t)
{
  return threadIdx.y;
}

/// Returns the thread index in the z direction.
fluidity_device_only inline std::size_t thread_id_impl(dimz_t)
{
  return threadIdx.z;
}

/// Returns the thread index in the \p dim dimension.
/// \param[in] dim The dimension to get the thread index for.
fluidity_device_only inline std::size_t thread_id_impl(std::size_t dim)
{
  return dim == dimx_t::value ? thread_id_impl(dim_x) :
         dim == dimy_t::value ? thread_id_impl(dim_y) :
         dim == dimz_t::value ? thread_id_impl(dim_z) : 0;
}

/// Returns the block index in the x direction.
fluidity_device_only inline std::size_t block_id_impl(dimx_t)
{
  return blockIdx.x;
}

/// Returns the thread index in the y direction.
fluidity_device_only inline std::size_t block_id_impl(dimy_t)
{
  return blockIdx.y;
}

/// Returns the block index in the z direction.
fluidity_device_only inline std::size_t block_id_impl(dimz_t)
{
  return blockIdx.z;
}

/// Returns the block index in the \p dim dimension.
/// \param[in] dim The dimension to get the block index for.
fluidity_device_only inline std::size_t block_id_impl(std::size_t dim)
{
  return dim == dimx_t::value ? block_id_impl(dim_x) :
         dim == dimy_t::value ? block_id_impl(dim_y) :
         dim == dimz_t::value ? block_id_impl(dim_z) : 0;
}

#endif // __CUDACC__  

}} // namespace fluid::detail

#endif // FLUIDITY_DIMENSION_THREAD_INDEX_DETAIL_HPP