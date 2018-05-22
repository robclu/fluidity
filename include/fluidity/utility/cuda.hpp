//==--- fluidity/utility/cuda_utils.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cuda_utils.hpp
/// \brief This file defines cuda utility functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_CUDA_UTILS_HPP
#define FLUIDITY_UTILITY_CUDA_UTILS_HPP

#include "cuda.cuh"
#include "debug.hpp"
#include "portability.hpp"
#include <fluidity/dimension/thread_index.hpp>

namespace fluid {
namespace util  {
namespace cuda  {

/// This namespace defines debugging functionality specifically for cuda.
namespace debug {

#if defined(__CUDACC__)

/// Prints the message \p msg, along with the thread and block information
/// for the thread.
template <typename Msg>
fluidity_device_only void thread_msg(Msg&& msg)
{
  printf("\n|====================================|"
         "\n| B,T: (%3lu, %3lu, %3lu),(%3lu,%3lu,%3lu) |"
         "\n|------------------------------------|"
         "\n|  %s"
         "\n|====================================|\n",
    block_id(dim_x) , block_id(dim_y) , block_id(dim_z) ,
    thread_id(dim_x), thread_id(dim_y), thread_id(dim_z),
    msg
  );
}

#else

/// Prints the message \p msg, along with the thread and block information
/// for the thread.
template <typename Msg>
fluidity_host_only void thread_msg(Msg&& msg) {}

#endif // __CUDACC__

} // namespace debug

/// Copies \p bytes of data from \p dev_ptr to \p dev_ptr.
/// \param[in]  dev_ptr_in   The device pointer to copy from.
/// \param[in]  dev_ptr_out  The device pointer to copy to.
/// \param[in]  bytes        The number of bytes to copy.
/// \tparam     DevPtr       The type of the device pointer.
template <typename DevPtr>
static inline void memcpy_device_to_device(const DevPtr* dev_ptr_in ,
                                           DevPtr*       dev_ptr_out,
                                           std::size_t   bytes      )
{
#if defined(__CUDACC__)
  constexpr auto num_threads = 2014;
  const     auto elements    = bytes / sizeof(DevPtr);
  auto threads = dim3(num_threads);
  auto blocks  = dim3(elements / num_threads);

  copy<<<blocks, threads>>>(dev_ptr_in, dev_ptr_out, elements);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

/// Copies \p bytes of data from \p host_ptr to \p dev_ptr.
/// \param[in]  host_ptr The host pointer to copy from.
/// \param[in]  dev_ptr  The device pointer to copy to.
/// \param[in]  bytes    The number of bytes to copy.
/// \tparam     HostPtr  The type of the host pointer.
/// \tparam     DevPtr   The type of the device pointer.
template <typename HostPtr, typename DevPtr>
static inline void memcpy_host_to_device(const HostPtr* host_ptr,
                                         DevPtr*        dev_ptr ,
                                         std::size_t    bytes   )
{
  fluidity_check_cuda_result(
    cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice)
  );
}

/// Copies \p bytes of data from \p dev_ptr to \p host_ptr.
/// \param[in]  dev_ptr  The device pointer to copy to.
/// \param[in]  host_ptr The host pointer to copy from.
/// \param[in]  bytes    The number of bytes to copy.
/// \tparam     DevPtr   The type of the device pointer.
/// \tparam     HostPtr  The type of the host pointer.
template <typename DevPtr, typename HostPtr>
static inline void memcpy_device_to_host(const DevPtr* dev_ptr ,
                                         HostPtr*      host_ptr,
                                         std::size_t   bytes   )
{
  fluidity_check_cuda_result(
    cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)
  );
}

/// Allocates \p bytes of memory on the device pointer to by \p dev_ptr.
/// \param[in] dev_ptr The device pointer to allocate memory for.
/// \param[in] bytes   The number of bytes to allocate.
/// \tparam    Ptr     The type of the pointer.
template <typename Ptr>
static inline void allocate(Ptr** dev_ptr, std::size_t bytes) 
{
  fluidity_check_cuda_result(cudaMalloc((void**)dev_ptr, bytes));
}

/// Frees the pointer \p ptr.
/// \param[in] ptr The pointer to free.
/// \tparam    Ptr The type of the pointer to free.
template <typename Ptr>
static inline void free(Ptr* ptr)
{
  fluidity_check_cuda_result(cudaFree(ptr));
}

}}} // namespace fluid::ciuda::util

#endif // FLUIDITY_UTILITY_CUDA_UTILS_HPP