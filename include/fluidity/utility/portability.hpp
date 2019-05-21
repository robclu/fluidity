//==--- fluidity/utility/portability.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  portability.hpp
/// \brief This file defines utilities for portability.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_PORTABILITY_HPP
#define FLUIDITY_UTILITY_PORTABILITY_HPP

#include <cuda.h>
#include <cuda_runtime.h>

/// Definitions for host, device, and host device functions, if CUDA is
/// supported.
#if defined(__CUDACC__) || defined(__clang__)
  #define fluidity_host_only   __host__
  #define fluidity_device_only __device__
  #define fluidity_host_device __host__ __device__ 
  #define fluidity_global      __global__

  /// Macro for thread synchronization for the device.
  #define fluidity_syncthreads() __syncthreads()
#else
  #define fluidity_host_only
  #define fluidity_device_only
  #define fluidity_host_device
  #define fluidity_global

  /// Macro for thread synchronization for the host.
  #define fluidity_syncthreads() 
#endif

#endif // FLUIDITY_UTILITY_PORTABILITY_HPP
