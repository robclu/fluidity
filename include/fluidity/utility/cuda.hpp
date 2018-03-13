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

#include "debug.hpp"
#include "portability.hpp"

namespace fluid {
namespace util  {
namespace cuda  {

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