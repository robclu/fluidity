//==--- fluidity/algorithm/cuda/fast_iterative_method.cuh -- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  broadcast.hpp
/// \brief This file provides a cuda implementation for warp and block
///        broadcasting on the device.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_CUDA_BROADCAST_CUH
#define FLUIDITY_ALGORITHM_CUDA_BROADCAST_CUH

#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace cuda  { 

/// Performs a reduction sum over the entire warp using the value \p val for
/// each of the threads. Each thread gets the result of the sum.
/// \param[in] val    The value to add to the reduction for this thread.
/// \tparam    T      The type of the value.
template <typename T>
fluidity_device_only T warp_broadcast_sum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
};

/// Performs a reduction sum over the entire block using the value \p val for
/// each of the threads. The result of the sum is returned to all threads in the
/// block.
/// \param[in] val    The value to add to the reduction for this thread.
/// \param[in] shared A pointer to the start of a shared memory block to use.
///                   This should be at least the size of the warp.
/// \tparam    T      The type of the value.
template <typename T>
fluidity_device_only T block_broadcast_sum(T val, T* shared) {
  const auto flat_idx = static_cast<int>(flattened_thread_id());
  const auto lane     = flat_idx % warpSize;
  const auto wid      = flat_idx / warpSize;
  const auto warps    = static_cast<std::size_t>(
    std::ceil(
      static_cast<float>(block_size()) / 
      static_cast<float>(warpSize)
    )
  );
        
  // Compute the warp sum, sending the result to all threads in the warp.
  val = warp_broadcast_sum(val);

  // Load the value into shared memory. Here, we need to consider the case that
  // there are less warps than the size of the shared memory, hence the else
  // branch. This is easier and faster than passing the shared memory size to
  // the kernel.
  if (lane == 0) {
    shared[wid] = val;
  } else if (flat_idx >= warps && flat_idx < warpSize) {
    shared[flat_idx] = T{0};
  }
  __syncthreads();

  // Reload the results back from the shared memory, and then perform another
  // broadcast sum of all warp sums to send the result to all threads.
  val = shared[lane];

  return warp_broadcast_sum(val);
}

}}

#endif // FLUIDITY_ALGORITHM_CUDA_BROADCAST_CUH
